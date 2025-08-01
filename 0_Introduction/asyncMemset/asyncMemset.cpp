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

#define DATA_SIZE (1 << 20) // 1M

static void memsetAsync()
{
    char *A_h = nullptr;
    char *A_d = nullptr;
    int memsetval = 0x1D;

    size_t numElements = DATA_SIZE;
    size_t num_of_bytes = numElements * sizeof(char);
    bool result = true;
    constexpr auto max_offset = 3; /* To memset on unaligned ptr. */
    mcStream_t stream;

    A_h = reinterpret_cast<char *>(malloc(num_of_bytes));
    if(A_h == nullptr)
    {
        return;
    }

    mcStreamCreate(&stream);
    mcMalloc(reinterpret_cast<void **>(&A_d), num_of_bytes);

    for (int offset = max_offset; offset >= 0; offset--)
    {
        mcMemsetAsync(A_d + offset, memsetval, numElements - 2 * offset, stream);
        mcMemsetAsync(A_d, memsetval - 1, offset, stream);
        mcMemsetAsync(A_d + numElements - offset, memsetval + 1, offset, stream);
        mcStreamSynchronize(stream);
        mcMemcpy(A_h, A_d, num_of_bytes, mcMemcpyDeviceToHost);
        for (size_t i = offset; i < numElements - offset; i++)
        {
            if (A_h[i] != memsetval)
            {
                result = false;
                break;
            }
        }
        for (size_t i = 0; i < offset; i++)
        {
            if (A_h[i] != memsetval - 1)
            {
                result = false;
                break;
            }
        }
        for (size_t i = numElements - offset; i < numElements; i++)
        {
            if (A_h[i] != memsetval + 1)
            {
                result = false;
                break;
            }
        }
        if(false == result)
        {
            break;
        }
    }

    mcFree(A_d);
    mcStreamDestroy(stream);
    free(A_h);

    if (result) {
        printf("memsetAsync: success!\n");
    } else {
        printf("memsetAsync: fail!\n");
    }

}

int main()
{
    memsetAsync();

    return 0;
}