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
 * Dynamic Paralleslism
 *
 * This is a very basic dynamic parallelism sample,
 * which shows how to use MACA Runtime API provided by the MACA dynamic paralleism.
 */

#include <stdio.h>
#include <mc_runtime.h>

__global__ void partial_sum(int *output, int *input, int i)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (0 == i)
  {
    output[tid] = 0;
  }
  for (int b = 0; b < i; b++)
  {
    output[i] += input[b];
  }
}

__global__ void dynamic_parallelism_kernel(int *src, int *dst)
{
  int tid = blockDim.x * blockIdx.x + threadIdx.x;

  dst[tid] = 1;
  partial_sum<<<1, 1>>>(dst, src, tid);
}

void verify_result(int *input, int *output, int n)
{
  for (int i = 0; i < n; i++)
  {
    int expected = 1;
    if (0 == i)
    {
      expected = 0;
    }
    for (int j = 0; j < i; j++)
    {
      expected += input[j];
    }
    if (output[i] != expected)
    {
      printf("dynamic parallelism test failed: index = %d, output = %d, expected = %d\n",
             i, output[i], expected);
    }
  }

  printf("dynamic parallelism test passed\n");
}

int main(void)
{
  int *input_h, *output_h;
  int *input_d, *output_d;
  int num_blocks, num_threads_per_block, test_num;

  num_blocks = 8;
  num_threads_per_block = 128;
  test_num = num_blocks * num_threads_per_block;

  input_h = (int *)malloc(test_num * sizeof(int));
  output_h = (int *)malloc(test_num * sizeof(int));
  mcMalloc((void **)&input_d, test_num * sizeof(int));
  mcMalloc((void **)&output_d, test_num * sizeof(int));
  for (int i = 0; i < test_num; i++)
  {
    input_h[i] = i;
  }

  mcMemcpy(input_d, input_h, test_num * sizeof(int), mcMemcpyHostToDevice);
  dynamic_parallelism_kernel<<<num_blocks, num_threads_per_block>>>(input_d, output_d);
  mcMemcpy(output_h, output_d, test_num * sizeof(int), mcMemcpyDeviceToHost);

  verify_result(input_h, output_h, test_num);

  mcFree(input_d);
  mcFree(output_d);

  free(input_h);
  free(output_h);

  return 0;
}
