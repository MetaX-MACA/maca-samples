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

#include <math.h>
#include <mc_runtime.h>

#define BLOCK_SIZE 16
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct
{
    int width;
    int height;
    int stride;
    float *elements;
} Matrix2;
// Get a matrix element
__device__ float GetElement(const Matrix2 A, int row, int col)
{
    return A.elements[row * A.stride + col];
}
// Set a matrix element
__device__ void SetElement(Matrix2 A, int row, int col, float value)
{
    A.elements[row * A.stride + col] = value;
}
// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
__device__ Matrix2 GetSubMatrix(Matrix2 A, int row, int col)
{
    Matrix2 Asub;
    Asub.width = BLOCK_SIZE;
    Asub.height = BLOCK_SIZE;
    Asub.stride = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col];
    return Asub;
}

// Matrix multiplication kernel called by MatMul_WithSharedMemory()
__global__ void MatMulKernel_WithSharedMemory(Matrix2 A, Matrix2 B, Matrix2 C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrix2 Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m)
    {
        // Get sub-matrix Asub of A
        Matrix2 Asub = GetSubMatrix(A, blockRow, m);
        // Get sub-matrix Bsub of B
        Matrix2 Bsub = GetSubMatrix(B, m, blockCol);
        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
}

int main()
{
    mcError_t err = mcSuccess;
    int m = 128;
    int n = 256;
    int k = 512;
    // int m = 16;
    // int n = 32;
    // int k = 16;
    Matrix2 A{n, m}, B{k, n}, C{k, m};
    mcMallocHost(&A.elements, m * n * sizeof(float));
    mcMallocHost(&B.elements, n * k * sizeof(float));
    mcMallocHost(&C.elements, m * k * sizeof(float));

    auto init_host = [](int row, int col, float *data)
    {
        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                data[i * col + j] = rand() / (float)RAND_MAX;
            }
        }
    };
    init_host(m, n, A.elements);
    init_host(n, k, B.elements);

    // Load A and B to device memory
    Matrix2 d_A;
    d_A.width = d_A.stride = A.width;
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    mcMalloc(&d_A.elements, size);
    mcMemcpy(d_A.elements, A.elements, size, mcMemcpyHostToDevice);

    Matrix2 d_B;
    d_B.width = d_B.stride = B.width;
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    mcMalloc(&d_B.elements, size);
    mcMemcpy(d_B.elements, B.elements, size, mcMemcpyHostToDevice);

    Matrix2 d_C;
    d_C.width = d_C.stride = C.width;
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    mcMalloc(&d_C.elements, size);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel_WithSharedMemory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    mcStreamSynchronize(0);
    err = mcGetLastError();

    if (err != mcSuccess)
    {
        fprintf(stderr, "Failed to launch MatMulKernel_WithSharedMemory kernel (error code %s)!\n",
                mcGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    mcMemcpy(C.elements, d_C.elements, size, mcMemcpyDeviceToHost);

    // Verify that the result vector is correct
    for (int rol = 0; rol < m; ++rol)
    {
        for (int col = 0; col < k; ++col)
        {
            float host_ele = 0.0f;
            int index = rol * k + col;
            for (int cnt = 0; cnt < n; cnt++)
            {
                host_ele += A.elements[rol * n + cnt] * B.elements[cnt * k + col];
            }

            if (fabs(C.elements[index] - host_ele) > 1e-3)
            {
                fprintf(stderr, "Result verification failed at element %d!\n", index);
                exit(EXIT_FAILURE);
            }
        }
    }
    mcFreeHost(A.elements);
    mcFreeHost(B.elements);
    mcFreeHost(C.elements);
    mcFree(d_A.elements);
    mcFree(d_B.elements);
    mcFree(d_C.elements);

    printf("Test passed for matrix multiplication kernel with shared memory!\n");
}
