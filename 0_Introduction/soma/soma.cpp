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

#include <mc_runtime.h>
#include <iostream>
#include <string.h>

template <typename T>
unsigned setNumBlocks(T blocksPerAP, T threadsPerBlock, size_t N)
{
    int device;
    mcGetDevice(&device);
    mcDeviceProp_t props;
    mcGetDeviceProperties(&props, device);

    unsigned blocks = props.multiProcessorCount * blocksPerAP;
    if (blocks * threadsPerBlock > N)
    {
        blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    }
    return blocks;
}

template <typename T>
__global__ void vectorADD(const T *A_d, const T *B_d, T *C_d, size_t NELEM)
{
    size_t offset = (blockIdx.x * blockDim.x + threadIdx.x);
    size_t stride = blockDim.x * gridDim.x;

    for (size_t i = offset; i < NELEM; i += stride)
    {
        C_d[i] = A_d[i] + B_d[i];
    }
}

bool checkVectorADD(int *A_h, int *B_h, int *C_h, size_t data_num)
{
    for (size_t i = 0; i < data_num; i++)
    {
        if (C_h[i] != A_h[i] + B_h[i])
        {
            return false;
        }
    }
    return true;
}

static unsigned g_blocks{6};
static unsigned g_threads{256};

int main()
{
    size_t data_num = 2 * 1024;
    size_t data_bytes = data_num * sizeof(int);
    int *A_d{nullptr}, *B_d{nullptr}, *C_d{nullptr};
    int *A_h{nullptr}, *B_h{nullptr}, *C_h{nullptr};

    A_h = (int *)malloc(data_bytes);
    B_h = (int *)malloc(data_bytes);
    C_h = (int *)malloc(data_bytes);
    memset(A_h, 3, data_num);
    memset(B_h, 4, data_num);
    memset(C_h, 0, data_num);

    mcStream_t stream;

    mcMemPool_t memPool;
    int device = 0;
    int deviceSupportsMemoryPools = 0;
    mcDeviceGetAttribute(&deviceSupportsMemoryPools, mcDeviceAttributeMemoryPoolsSupported,
                         device);
    if (deviceSupportsMemoryPools == 0)
    {
        printf("device do not support the Stream Ordered MemoryAllocator.\n");
        return 0;
    }

    mcSetDevice(device);
    mcStreamCreate(&stream);

    /* the mempool from mcMemPoolCreate */
    mcMemPoolProps poolProps;
    mcGetDevice(&device);

    memset(&poolProps, 0, sizeof(mcMemPoolProps));
    poolProps.allocType = mcMemAllocationTypePinned;
    poolProps.handleTypes = mcMemHandleTypeNone;
    poolProps.location.id = device;
    poolProps.location.type = mcMemLocationTypeDevice;
    mcMemPoolCreate(&memPool, &poolProps);

    mcMallocFromPoolAsync((void **)&A_d, data_bytes, memPool, stream);
    mcMallocFromPoolAsync((void **)&B_d, data_bytes, memPool, stream);
    mcMallocFromPoolAsync((void **)&C_d, data_bytes, memPool, stream);

    mcMemcpyAsync(A_d, A_h, data_bytes, mcMemcpyHostToDevice, stream);
    mcMemcpyAsync(B_d, B_h, data_bytes, mcMemcpyHostToDevice, stream);

    unsigned blocks = setNumBlocks(g_blocks, g_threads, data_num);

    vectorADD<<<dim3(blocks), dim3(g_threads), 0, stream>>>(A_d, B_d, C_d,
                                                            data_num);

    mcMemcpyAsync(C_h, C_d, data_bytes, mcMemcpyDeviceToHost, stream);

    mcFreeAsync(A_d, stream);
    mcFreeAsync(B_d, stream);
    mcFreeAsync(C_d, stream);

    mcStreamSynchronize(stream);

    bool res = checkVectorADD(A_h, B_h, C_h, data_num);
    if (!res)
    {
        perror("case Failed!\n");
        return 0;
    }

    mcStreamDestroy(stream);
    mcMemPoolDestroy(memPool);

    free(A_h);
    free(B_h);
    free(C_h);
    printf("case pass!\n");

    return 0;
}
