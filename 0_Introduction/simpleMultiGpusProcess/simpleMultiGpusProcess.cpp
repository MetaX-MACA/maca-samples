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

#include <iostream>
#include <fstream>
#include <iostream>
#include <cmath>
#include <cmath>
#include <mc_runtime.h>
#include <mc_common.h>
#include <mpi.h>

#define MAX_NUM_STREAM 10
#define STREAMS_PER_DEVICE 2
#define TOTAL_STREAMS 4
#define THREAD_NUM_PER_BLOCK 32
#define BLOCK_NUM 32

template <typename T>
__global__ void vectoradd_kernel(const T *__restrict__ a, const T *__restrict__ b,
                                 T *__restrict__ c, int width, int loop)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int i = y * width + x;
    for (size_t j = 0; j < loop; j++)
    {
        if (i < (width))
        {
            c[i] = a[i] + b[i];
        }
    }
}
template <typename T>
static void BM_data_parallelism()
{
    T *hostA[MAX_NUM_STREAM];
    T *hostB[MAX_NUM_STREAM];
    T *hostC[MAX_NUM_STREAM];

    T *deviceA[MAX_NUM_STREAM];
    T *deviceB[MAX_NUM_STREAM];
    T *deviceC[MAX_NUM_STREAM];

    mcStream_t streams[MAX_NUM_STREAM];
    int streams_per_device = STREAMS_PER_DEVICE;
    int run_streams = TOTAL_STREAMS;
    int num_blocks = BLOCK_NUM;
    int num_threads_per_block = THREAD_NUM_PER_BLOCK;
    int loop_times = 1;
    int buffer_size = num_blocks * num_threads_per_block;

    printf("streams_per_device:%d, total streams:%d, blocks = %d, threads_per_block = %d, loop=%d\n", streams_per_device,
           run_streams, num_blocks, num_threads_per_block, loop_times);
    if (run_streams >= MAX_NUM_STREAM)
    {
        return;
    }
    int myRank, nRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
    MPI_Comm_size(MPI_COMM_WORLD, &nRanks);

    int counts = 0;
    mcGetDeviceCount(&counts);
    printf("device count:%d\n", counts);
    if ((streams_per_device > run_streams)||(counts *streams_per_device < run_streams))
    {
        printf("please check config parameters and device count is correct!\n");
        return;
    }

    int errors = 0;

    // Prepare host and device memory
    for (int i = 0; i < run_streams; ++i)
    {
        mcSetDevice(i / streams_per_device); // three streams per device
        hostA[i] = (T *)malloc(buffer_size * sizeof(T));
        hostB[i] = (T *)malloc(buffer_size * sizeof(T));
        hostC[i] = (T *)malloc(buffer_size * sizeof(T));
        // initialize the input data
        for (unsigned int j = 0; j < buffer_size; j++)
        {
            hostA[i][j] = 0xff;
            hostB[i][j] = 0x1;
        }
        mcMalloc((void **)&deviceA[i], buffer_size * sizeof(T));
        mcMalloc((void **)&deviceB[i], buffer_size * sizeof(T));
        mcMalloc((void **)&deviceC[i], buffer_size * sizeof(T));

        // Create streams for concurrency
        mcStreamCreate(&streams[i]);
        mcMemcpyAsync(deviceA[i], hostA[i], buffer_size * sizeof(T), mcMemcpyHostToDevice,
                      streams[i]);
        mcMemcpyAsync(deviceB[i], hostB[i], buffer_size * sizeof(T), mcMemcpyHostToDevice,
                      streams[i]);

        vectoradd_kernel<T>
            <<<dim3(num_blocks, 1), dim3(num_threads_per_block, 1), 0, streams[i]>>>(
                deviceA[i], deviceB[i], deviceC[i], buffer_size, loop_times);
    }

    for (int i = 0; i < run_streams; ++i)
    {
        mcSetDevice(i / streams_per_device); // three streams per device
        // Synchronize all the concurrent streams to have completed execution
        mcMemcpyAsync(hostC[i], deviceC[i], buffer_size * sizeof(T), mcMemcpyDeviceToHost,
                      streams[i]);
        mcStreamSynchronize(streams[i]);
        errors = 0;
        // verify the results
        for (int j = 0; j < buffer_size; j++)
        {
            T expected = hostA[i][j] + hostB[i][j];
            T out = hostC[i][j];
            if (fabs(out - expected) > 0.000001)
            {
                errors++;
            }
        }
        mcFree(deviceA[i]);
        mcFree(deviceB[i]);
        mcFree(deviceC[i]);

        mcStreamDestroy(streams[i]);

        free(hostA[i]);
        free(hostB[i]);
        free(hostC[i]);
    }
    if (errors != 0)
    {
        printf("%s FAILED: %d errors\n", __func__, errors);
    }
    else
    {
        printf("total process:%d, process:%d, %s PASSED!\n", nRanks, myRank, __func__);
    }
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    BM_data_parallelism<float>();
    MPI_Finalize();
}