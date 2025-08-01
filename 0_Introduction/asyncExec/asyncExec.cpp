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

#include <cstdio>
#include <cmath>
#include <mc_runtime.h>

#define NUM_ELEMENTS 4096
#define BLOCK_SIZE   256
#define ITERATIONS   20


template <typename T> __global__ void kernel_add(T *d_A, int numElements, T addend)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        d_A[i] += addend;
    }
}

int main(void) {
    int numElements   = NUM_ELEMENTS;
    int blockSize     = BLOCK_SIZE;
    int iterations    = ITERATIONS;
    int numBlocks     = (numElements + blockSize - 1) / blockSize;
    float expectValue = static_cast<float>(iterations) * (iterations - 1) / 2;
    float *h_A, *d_A;

    mcMallocHost(&h_A, sizeof(float) * numElements);
    for (int i = 0; i < numElements; i++) {
        h_A[i] = 0.0f;
    }

    for (int i = 0; i < iterations; i++) {
        mcMallocAsync(reinterpret_cast<void **>(&d_A), sizeof(float) * numElements, NULL);
        mcMemcpyAsync(d_A, h_A, sizeof(float) * numElements, mcMemcpyHostToDevice,
                        NULL);
        kernel_add<float><<<numBlocks, blockSize>>>(d_A, numElements, static_cast<float>(i));
        mcMemcpyAsync(h_A, d_A, sizeof(float) * numElements, mcMemcpyDeviceToHost, NULL);
        mcFreeAsync(d_A, NULL);
    }
    mcStreamSynchronize(NULL);

    for (int i = 0; i < numElements; i++) {
        if (fabs(h_A[i] - expectValue) > 1e-3)
        {
            fprintf(stderr, "Result verification failed at element %d, expect %f, actually %f!\n", i, expectValue, h_A[i]);
            exit(EXIT_FAILURE);
        }
    }
    printf("Result verification passed!\n");
    mcFreeHost(h_A);
    return 0;
}
