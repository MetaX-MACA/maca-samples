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

#include <string.h>
#include <mc_runtime.h>
#include <stdlib.h>

__global__ void vectorAdditionWithProfiling(const float *a, const float *b, float *c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

__global__ void vectorAdditionWithoutProfiling(const float *a, const float *b, float *c, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    const int vectorSize = 5;
    // host memory
    float h_a[vectorSize] = {1.0, 2.0, 3.0, 4.0, 5.0};
    float h_b[vectorSize] = {5.0, 4.0, 3.0, 2.0, 1.0};
    float h_c[vectorSize];
    // device memory
    float *d_a, *d_b, *d_c;
    mcMalloc((void **)&d_a, vectorSize * sizeof(float));
    mcMalloc((void **)&d_b, vectorSize * sizeof(float));
    mcMalloc((void **)&d_c, vectorSize * sizeof(float));
    // copy data from host to device
    mcMemcpy(d_a, h_a, vectorSize * sizeof(float), mcMemcpyHostToDevice);
    mcMemcpy(d_b, h_b, vectorSize * sizeof(float), mcMemcpyHostToDevice);
    int blockSize = 256;
    int gridSize = (vectorSize + blockSize - 1) / blockSize;
    // launch kernel before mcProfilerStart
    vectorAdditionWithoutProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    // first start
    mcProfilerStart();
    // launch 3 kernels between start and stop
    vectorAdditionWithProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    vectorAdditionWithProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    vectorAdditionWithProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    mcProfilerStop();
    // launch kernel, not in profiler scope
    vectorAdditionWithoutProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    // second start
    mcProfilerStart();
    // launch 2 kernels between start and stop
    vectorAdditionWithProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    vectorAdditionWithProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    mcProfilerStop();
    // launch 2 kernels, not in profiler scope
    vectorAdditionWithoutProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);
    vectorAdditionWithoutProfiling<<<gridSize, blockSize>>>(d_a, d_b, d_c, vectorSize);

    mcFree(d_a);
    mcFree(d_b);
    mcFree(d_c);
    return 0;
}