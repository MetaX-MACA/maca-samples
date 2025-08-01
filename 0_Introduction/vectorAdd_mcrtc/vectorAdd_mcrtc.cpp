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

/**
 * Vector addition: C = A + B.
 */

#include <stdio.h>
#include <cmath>
#include <mc_runtime.h>
#include <mcrtc_helper.h>

/**
 * Host main routine
 */
int main(int argc, char **argv)
{
    char *bitcode;
    size_t bitcodeSize;
    char kernel_file[] = "vectorAdd_kernel.maca";
    compileFileToBitcode(kernel_file, argc, argv, &bitcode, &bitcodeSize, 0);
    mcModule_t module = loadCode(bitcode, argc, argv);

    mcFunction_t kernel_addr;
    mcModuleGetFunction(&kernel_addr, module, "vectorAdd");

    // Print the vector length to be used, and compute its size
    int numElements = 50000;
    size_t size = numElements * sizeof(float);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    float *h_A = reinterpret_cast<float *>(malloc(size));

    // Allocate the host input vector B
    float *h_B = reinterpret_cast<float *>(malloc(size));

    // Allocate the host output vector C
    float *h_C = reinterpret_cast<float *>(malloc(size));

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand() / static_cast<float>(RAND_MAX);
        h_B[i] = rand() / static_cast<float>(RAND_MAX);
    }

    // Allocate the device input vector A
    float *d_A;
    mcMalloc(&d_A, size);

    // Allocate the device input vector B
    float *d_B;
    mcMalloc(&d_B, size);

    // Allocate the device output vector C
    float *d_C;
    mcMalloc(&d_C, size);

    // Copy the host input vectors A and B in host memory to the device input
    // vectors in device memory
    printf("Copy input data from the host memory to the MC device\n");

    mcMemcpy(d_A, h_A, size, mcMemcpyHostToDevice);
    mcMemcpy(d_B, h_B, size, mcMemcpyHostToDevice);

    // Launch the Vector Add GPU Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    printf("GPU kernel launch with %d blocks of %d threads\n", blocksPerGrid,
           threadsPerBlock);
    dim3 mcBlockSize(threadsPerBlock, 1, 1);
    dim3 mcGridSize(blocksPerGrid, 1, 1);

    void *arr[] = {reinterpret_cast<void *>(&d_A), reinterpret_cast<void *>(&d_B),
                   reinterpret_cast<void *>(&d_C),
                   reinterpret_cast<void *>(&numElements)};
    mcModuleLaunchKernel(kernel_addr, mcGridSize.x, mcGridSize.y,
                         mcGridSize.z, /* grid dim */
                         mcBlockSize.x, mcBlockSize.y,
                         mcBlockSize.z, /* block dim */
                         0, 0,          /* shared mem, stream */
                         &arr[0],       /* arguments */
                         0);
    mcDeviceSynchronize();

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the MC device to the host memory\n");
    mcMemcpy(h_C, d_C, size, mcMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Free device global memory
    mcFree(d_A);
    mcFree(d_B);
    mcFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");

    return 0;
}
