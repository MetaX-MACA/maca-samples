/* Copyright © 2023 MetaX Integrated Circuits (Shanghai) Co.,Ltd. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are
 * permitted provided that the following conditions are met:
 *  1. Redistributions of source code must retain the above copyright notice, this list of
 *     conditions and the following disclaimer.
 *  2. Redistributions in binary form must reproduce the above copyright notice, this list of
 *     conditions and the following disclaimer in the documentation and/or other materials
 *     provided with the distribution.
 *  3. Neither the name of the copyright holder nor the names of its contributors may be used
 *     to endorse or promote products derived from this software without specific prior written
 *     permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This sample demonstrates Inter Process Communication
 * using one process per GPU for computation.
 */
#include <stdio.h>
#include <stdlib.h>
#include <vector>
// For the MACA runtime routines
#include <mc_runtime.h>
#include <mc_common.h>

#include "helper_maca.h"
#include "helper_multiprocess.h"
static const char shmName[] = "simpleIPCshm";

#define MAX_DEVICES (32)
#define DATA_SIZE (64ULL << 20ULL) // 64MB

#if defined(__linux__)
#define cpu_atomic_add32(a, x) __sync_add_and_fetch(a, x)
#elif defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define cpu_atomic_add32(a, x) InterlockedAdd((volatile LONG *)a, x)
#else
#error Unsupported system
#endif

typedef struct shmStruct_st
{
  size_t nprocesses;
  int barrier;
  int sense;
  int devices[MAX_DEVICES];
  mcIpcMemHandle_t memHandle[MAX_DEVICES];
  mcIpcEventHandle_t eventHandle[MAX_DEVICES];
} shmStruct;

__global__ void simpleKernel(char *ptr, int sz, char val)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  for (; idx < sz; idx += (gridDim.x * blockDim.x))
  {
    ptr[idx] = val;
  }
}

static void barrierWait(volatile int *barrier, volatile int *sense,
                        unsigned int n)
{
  int count;

  // Check-in
  count = cpu_atomic_add32(barrier, 1);
  if (count == n) // Last one in
    *sense = 1;
  while (!*sense)
    ;

  // Check-out
  count = cpu_atomic_add32(barrier, -1);
  if (count == 0) // Last one out
    *sense = 0;
  while (*sense)
    ;
}

static void childProcess(int id)
{
  volatile shmStruct *shm = NULL;
  mcStream_t stream;
  sharedMemoryInfo info;
  size_t procCount, i;
  int blocks = 0;
  int threads = 128;
  mcDeviceProp_t prop;
  std::vector<void *> ptrs;
  std::vector<mcEvent_t> events;
  std::vector<char> verification_buffer(DATA_SIZE);

  if (sharedMemoryOpen(shmName, sizeof(shmStruct), &info) != 0)
  {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  procCount = shm->nprocesses;

  printf("Process %d: Starting on device %d...\n", id, shm->devices[id]);

  checkMacaErrors(mcSetDevice(shm->devices[id]));
  checkMacaErrors(mcGetDeviceProperties(&prop, shm->devices[id]));
  checkMacaErrors(mcStreamCreateWithFlags(&stream, mcStreamNonBlocking));
  checkMacaErrors(mcOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, simpleKernel, threads, 0));
  blocks *= prop.multiProcessorCount;

  // Open and track all the allocations and events created in the master
  // process for use later
  for (i = 0; i < procCount; i++)
  {
    void *ptr = NULL;
    mcEvent_t event;

    // Notice, we don't need to explicitly enable peer access for
    // allocations on other devices.
    checkMacaErrors(
        mcIpcOpenMemHandle(&ptr, *(mcIpcMemHandle_t *)&shm->memHandle[i],
                           mcIpcMemLazyEnablePeerAccess));
    checkMacaErrors(mcIpcOpenEventHandle(
        &event, *(mcIpcEventHandle_t *)&shm->eventHandle[i]));

    ptrs.push_back(ptr);
    events.push_back(event);
  }

  // At each iteration of the loop, each sibling process will push work on
  // their respective devices accessing the next peer mapped buffer allocated
  // by the master process (these can come from other sibling processes as
  // well). To coordinate each process' access, we force the stream to wait for
  // the work already accessing this buffer asynchronously through IPC events,
  // allowing the CPU processes to continue to queue more work.
  for (i = 0; i < procCount; i++)
  {
    size_t bufferId = (i + id) % procCount;
    // Wait for the buffer to be accessed to be ready
    checkMacaErrors(mcStreamWaitEvent(stream, events[bufferId], 0));
    // Push a simple kernel on it
    simpleKernel<<<blocks, threads, 0, stream>>>((char *)ptrs[bufferId],
                                                 DATA_SIZE, id);
    checkMacaErrors(mcGetLastError());
    // Signal that this buffer is ready for the next consumer
    checkMacaErrors(mcEventRecord(events[bufferId], stream));
    // Wait for all my sibling processes to push this stage of their work
    // before proceeding to the next. This prevents siblings from racing
    // ahead and clobbering the recorded event or waiting on the wrong
    // recorded event.
    barrierWait(&shm->barrier, &shm->sense, (unsigned int)procCount);
    if (id == 0)
    {
      printf("Step %lld done\n", (unsigned long long)i);
    }
  }

  // Now wait for my buffer to be ready so I can copy it locally and verify it
  checkMacaErrors(mcStreamWaitEvent(stream, events[id], 0));
  checkMacaErrors(mcMemcpyAsync(&verification_buffer[0], ptrs[id], DATA_SIZE,
                                mcMemcpyDeviceToHost, stream));
  // And wait for all the queued up work to complete
  checkMacaErrors(mcStreamSynchronize(stream));

  printf("Process %d: verifying...\n", id);

  // The contents should have the id of the sibling just after me
  char compareId = (char)((id + 1) % procCount);
  for (unsigned long long j = 0; j < DATA_SIZE; j++)
  {
    if (verification_buffer[j] != compareId)
    {
      printf("Process %d: Verification mismatch at %lld: %d != %d\n", id, j,
             (int)verification_buffer[j], (int)compareId);
    }
  }

  // Clean up!
  for (i = 0; i < procCount; i++)
  {
    checkMacaErrors(mcIpcCloseMemHandle(ptrs[i]));
    checkMacaErrors(mcEventDestroy(events[i]));
  }

  checkMacaErrors(mcStreamDestroy(stream));

  printf("Process %d complete!\n", id);
}

static void parentProcess(char *app)
{
  sharedMemoryInfo info;
  int devCount, i;
  volatile shmStruct *shm = NULL;
  std::vector<void *> ptrs;
  std::vector<mcEvent_t> events;
  std::vector<Process> processes;

  checkMacaErrors(mcGetDeviceCount(&devCount));

  if (sharedMemoryCreate(shmName, sizeof(*shm), &info) != 0)
  {
    printf("Failed to create shared memory slab\n");
    exit(EXIT_FAILURE);
  }
  shm = (volatile shmStruct *)info.addr;
  memset((void *)shm, 0, sizeof(*shm));

  // Pick all the devices that can access each other's memory for this test
  // Keep in mind that MACA has minimal support for fork() without a
  // corresponding exec() in the child process, but in this case our
  // spawnProcess will always exec, so no need to worry.
  for (i = 0; i < devCount; i++)
  {
    bool allPeers = true;
    mcDeviceProp_t prop;
    checkMacaErrors(mcGetDeviceProperties(&prop, i));

    // MACA IPC is only supported on devices with unified addressing
    if (!prop.unifiedAddressing)
    {
      printf("Device %d does not support unified addressing, skipping...\n", i);
      continue;
    }
    // This sample requires two processes accessing each device, so we need
    // to ensure exclusive or prohibited mode is not set
    if (prop.computeMode != mcComputeModeDefault)
    {
      printf("Device %d is in an unsupported compute mode for this sample\n",
             i);
      continue;
    }

    for (int j = 0; j < shm->nprocesses; j++)
    {
      int canAccessPeerIJ, canAccessPeerJI;
      checkMacaErrors(
          mcDeviceCanAccessPeer(&canAccessPeerJI, shm->devices[j], i));
      checkMacaErrors(
          mcDeviceCanAccessPeer(&canAccessPeerIJ, i, shm->devices[j]));
      if (!canAccessPeerIJ || !canAccessPeerJI)
      {
        allPeers = false;
        break;
      }
    }
    if (allPeers)
    {
      // Enable peers here.  This isn't necessary for IPC, but it will
      // setup the peers for the device.  For systems that only allow 8
      // peers per GPU at a time, this acts to remove devices from CanAccessPeer
      for (int j = 0; j < shm->nprocesses; j++)
      {
        checkMacaErrors(mcSetDevice(i));
        checkMacaErrors(mcDeviceEnablePeerAccess(shm->devices[j], 0));
        checkMacaErrors(mcSetDevice(shm->devices[j]));
        checkMacaErrors(mcDeviceEnablePeerAccess(i, 0));
      }
      shm->devices[shm->nprocesses++] = i;
      if (shm->nprocesses >= MAX_DEVICES)
        break;
    }
    else
    {
      printf(
          "Device %d is not peer capable with some other selected peers, "
          "skipping\n",
          i);
    }
  }

  if (shm->nprocesses == 0)
  {
    printf("No MACA devices support IPC\n");
    exit(EXIT_WAIVED);
  }

  // Now allocate memory and an event for each process and fill the shared
  // memory buffer with the IPC handles to communicate
  for (i = 0; i < shm->nprocesses; i++)
  {
    void *ptr = NULL;
    mcEvent_t event;

    checkMacaErrors(mcSetDevice(shm->devices[i]));
    checkMacaErrors(mcMalloc(&ptr, DATA_SIZE));
    checkMacaErrors(
        mcIpcGetMemHandle((mcIpcMemHandle_t *)&shm->memHandle[i], ptr));
    checkMacaErrors(mcEventCreate(
        &event, mcEventDisableTiming | mcEventInterprocess));
    checkMacaErrors(mcIpcGetEventHandle(
        (mcIpcEventHandle_t *)&shm->eventHandle[i], event));

    ptrs.push_back(ptr);
    events.push_back(event);
  }

  // Launch the child processes!
  for (i = 0; i < shm->nprocesses; i++)
  {
    char devIdx[10];
    char *const args[] = {app, devIdx, NULL};
    Process process;

    SPRINTF(devIdx, "%d", i);

    if (spawnProcess(&process, app, args))
    {
      printf("Failed to create process\n");
      exit(EXIT_FAILURE);
    }

    processes.push_back(process);
  }

  // And wait for them to finish
  for (i = 0; i < processes.size(); i++)
  {
    if (waitProcess(&processes[i]) != EXIT_SUCCESS)
    {
      printf("Process %d failed!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // Clean up!
  for (i = 0; i < shm->nprocesses; i++)
  {
    checkMacaErrors(mcSetDevice(shm->devices[i]));
    checkMacaErrors(mcEventSynchronize(events[i]));
    checkMacaErrors(mcEventDestroy(events[i]));
    checkMacaErrors(mcFree(ptrs[i]));
  }

  sharedMemoryClose(&info);
}

int main(int argc, char **argv)
{
#if defined(__arm__) || defined(__aarch64__)
  printf("Not supported on ARM\n");
  return EXIT_WAIVED;
#else
  if (argc == 1)
  {
    parentProcess(argv[0]);
  }
  else
  {
    childProcess(atoi(argv[1]));
  }
  return EXIT_SUCCESS;
#endif
}
