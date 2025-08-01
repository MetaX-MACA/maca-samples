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

#define DATA_SIZE (64 * (1e3)) // 64 k

void deviceToHostTransfer(unsigned int memSize)
{
    unsigned char *h_idata = NULL;
    unsigned char *h_odata = NULL;

    /* allocate host memory */
    mcMallocHost((void **)&h_idata, memSize);
    mcMallocHost((void **)&h_odata, memSize);

    /* initialize the memory */
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        h_idata[i] = (unsigned char)(i & 0xff);
    }

    /* allocate device memory */
    unsigned char *d_idata;
    mcMalloc((void **)&d_idata, memSize);

    /* initialize the device memory */
    mcMemcpy(d_idata, h_idata, memSize, mcMemcpyHostToDevice);

    /* copy data from GPU to Host */
    mcMemcpyAsync(h_odata, d_idata, memSize, mcMemcpyDeviceToHost, 0);
    mcDeviceSynchronize();

    /* check result */
    bool result = true;
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        if (h_idata[i] != h_odata[i]) {
            result = false;
            break;
        }
    }

    mcFreeHost(h_idata);
    mcFreeHost(h_odata);
    mcFree(d_idata);

    if (result) {
        printf("deviceToHostTransfer: success!\n");
    } else {
        printf("deviceToHostTransfer: fail!\n");
    }
}

void hostToDeviceTransfer(unsigned int memSize)
{
    /* allocate host memory */
    unsigned char *h_odata = NULL;

    /* pinned memory mode - use special function to get OS-pinned memory */
    mcMallocHost((void **)&h_odata, memSize);

    /* initialize the memory */
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        h_odata[i] = (unsigned char)(i & 0xff);
    }

    /* allocate device memory */
    unsigned char *d_idata;
    mcMalloc((void **)&d_idata, memSize);

    /* copy host memory to device memory */
    mcMemcpyAsync(d_idata, h_odata, memSize, mcMemcpyHostToDevice, 0);
    mcDeviceSynchronize();

    /* check result */
    unsigned char *h_data = NULL;
    h_data                = (unsigned char *)malloc(memSize);
    mcMemcpyAsync(h_data, d_idata, memSize, mcMemcpyDeviceToHost, 0);
    mcDeviceSynchronize();

    bool result = true;
    for (unsigned int i = 0; i < memSize / sizeof(unsigned char); i++) {
        if (h_data[i] != h_odata[i]) {
            result = false;
            break;
        }
    }

    mcFreeHost(h_odata);
    mcFree(d_idata);

    if (result) {
        printf("hostToDeviceTransfer: success!\n");
    } else {
        printf("hostToDeviceTransfer: fail!\n");
    }
}

int main()
{
    int memorySize = DATA_SIZE;

    /* copy from device to host pinned memory */
    deviceToHostTransfer(memorySize);

    /* copy from host pinned memory to device */
    hostToDeviceTransfer(memorySize);
}
