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

#define BLOCK_SIZE 1024
#define DATA_SIZE 512 * 1024

template <class T>
__global__ void iota(T *addr, size_t N, int init_value = 0)
{
    auto id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < N)
        addr[id] = id + init_value;
}

bool func()
{
    char *host = nullptr;
    host = (char *)malloc(DATA_SIZE);
    memset(host, 0, DATA_SIZE);

    char *dev = nullptr;
    mcMalloc(&dev, DATA_SIZE);

    mcStream_t stream;
    mcStreamCreate(&stream);
    mcMemcpyAsync(dev, host, DATA_SIZE, mcMemcpyHostToDevice, stream);
    iota<<<(DATA_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>((char *)dev, DATA_SIZE);
    mcMemcpyAsync(host, dev, DATA_SIZE, mcMemcpyDeviceToHost, stream);
    mcStreamSynchronize(stream);

    bool ok = true;
    for (int i = 0; i < DATA_SIZE; i++)
    {
        if (host[i] != char(i))
        {
            ok = false;
            break;
        }
    }
    if (!ok)
    {
        return false;
    }

    free(host);
    mcFree(dev);
    mcStreamDestroy(stream);
    return true;
}

int main()
{
    bool res = true;
    res = func();
    if (!res)
    {
        printf("verify fail\n");
        return 1;
    }
    printf("case pass!\n");
    return 0;
}
