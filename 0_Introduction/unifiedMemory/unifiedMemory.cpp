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

#include <stdio.h>
#include <math.h>
// For the MACA runtime routines
#include <mc_runtime.h>
#include <mc_common.h>

/**
 * MACA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < numElements) {
    C[i] = A[i] + B[i] + 0.0f;
  }
}

/**
 * Host main routine
 */
int main(void) {
  // Error code to check return values for MACA calls
  mcError_t err = mcSuccess;

  // Print the vector length to be used, and compute its size
  int numElements = 50000;
  size_t size = numElements * sizeof(float);
  printf("[Vector addition of %d elements]\n", numElements);

  float *prt_A=NULL;
  float *prt_B=NULL;
  float *prt_C=NULL;

  // Allocate the memory
  err = mcMallocManaged(&prt_A, size);
  if (err != mcSuccess) {
    fprintf(stderr, "Failed to allocate unified memory vector A (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = mcMallocManaged(&prt_B, size);
  if (err != mcSuccess) {
    fprintf(stderr, "Failed to allocate unified memory vector B (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }
  err = mcMallocManaged(&prt_C, size);
  if (err != mcSuccess) {
    fprintf(stderr, "Failed to allocate unified memory vector C (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Initialize the host input vectors
  for (int i = 0; i < numElements; ++i) {
    prt_A[i] = rand() / (float)RAND_MAX;
    prt_B[i] = rand() / (float)RAND_MAX;
  }

  // Launch the Vector Add MACA Kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
  printf("MACA kernel launch with %d blocks of %d threads\n", blocksPerGrid,
         threadsPerBlock);
  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(prt_A, prt_B, prt_C, numElements);
  err = mcGetLastError();

  if (err != mcSuccess) {
    fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy the device result vector in device memory to the host result vector
  // in host memory.
  printf("Copy output data from the MACA device to the host memory\n");
  err = mcDeviceSynchronize();

  if (err != mcSuccess) {
    fprintf(stderr,
            "Failed to synchronize the device (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Verify that the result vector is correct
  for (int i = 0; i < numElements; ++i) {
    if (fabs(prt_A[i] + prt_B[i] - prt_C[i]) > 1e-5) {
      fprintf(stderr, "Result verification failed at element %d!\n", i);
      exit(EXIT_FAILURE);
    }
  }

  printf("Test PASSED\n");

  // Free device global memory
  err = mcFree(prt_A);
  if (err != mcSuccess) {
    fprintf(stderr, "Failed to free unified memory vector A (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = mcFree(prt_B);
  if (err != mcSuccess) {
    fprintf(stderr, "Failed to free unified memory vector B (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  err = mcFree(prt_C);
  if (err != mcSuccess) {
    fprintf(stderr, "Failed to free unified memory vector C (error code %s)!\n",
            mcGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  printf("MACA Sample Done\n");
  return 0;
}
