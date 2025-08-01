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
#include <fstream>
#include "mccl.h"
#include <math.h>
#include <string.h>

#define NUM_STREAMS 2
#define GRID_SIZE 2
#define BLOCK_SIZE 2
#define DATA_TYPE 7

template <typename T>
static void initBuff(int *devs, T **&sendbuff, T **&recvbuff, T **&t, mcStream_t *&s, int ranks,
                     int size)
{
    sendbuff = (T **)malloc(ranks * sizeof(T *));
    recvbuff = (T **)malloc(ranks * sizeof(T *));
    t        = (T **)malloc(ranks * sizeof(T *));
    s        = (mcStream_t *)malloc(ranks * sizeof(mcStream_t));

    for (int i = 0; i < ranks; ++i) {
        /* alloc host input array with ranks */
        t[i] = (T *)malloc(size * sizeof(T));

        /* init device data */
        mcSetDevice(devs[i]);
        mcMalloc((void **)(sendbuff + i), size * sizeof(T));
        mcMalloc((void **)(recvbuff + i), size * sizeof(T));
        mcMemset(sendbuff[i], 0, size * sizeof(T));
        mcMemset(recvbuff[i], 0, size * sizeof(T));

        /* create stream */
        mcStreamCreate(s + i);
    }
}

template <class T>
static void deinitBuff(int *devs, T **&sendbuff, T **&recvbuff, T **&t, mcStream_t *&s, int ranks)
{
    /* free buffers */
    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        mcFree(sendbuff[i]);
        mcFree(recvbuff[i]);
        mcStreamDestroy(s[i]);
        free(t[i]);
    }

    free(s);
    free(t);
    free(sendbuff);
    free(recvbuff);
}

template <class T> static int checkValue(T *t, int size, double value)
{
    int ret = 0;

    for (int j = 0; j < size; j++) {
        if (fabs((double)t[j] - value) > 0.000001f) {
            printf("Found wrong value:t[%d]\n", j);
            ret = -1;
            break;
        }
    }

    return ret;
}

mcclDataType_t matchMcclDataType(int value)
{
    mcclDataType_t inputType = mcclNumTypes;
    switch (value) {
    case 0:
        inputType = mcclInt8;
        break;
    case 1:
        inputType = mcclUint8;
        break;
    case 2:
        inputType = mcclInt32;
        break;
    case 3:
        inputType = mcclUint32;
        break;
    case 4:
        inputType = mcclInt64;
        break;
    case 5:
        inputType = mcclUint64;
        break;
    case 6:
        inputType = mcclFloat16;
        break;
    case 7:
        inputType = mcclFloat32;
        break;
    case 8:
        inputType = mcclFloat64;
        break;
    case 9:
        inputType = mcclBfloat16;
        break;
    default:
        break;
    }

    return inputType;
}

template <typename T> __global__ void data_parallel(T *d_A, T *out, int n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (0 == i) {
        *out += n * d_A[0];
    }
}

template <typename T> __global__ void data_parallel_B(T *d_A, T *out)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    atomicAdd(out, d_A[i]);
}

/*
 * Root GPU in data parallel is controller, which split the dataSet and gather compute
 * results to update weight and bias value in pytorch. By the way, Non-root GPUs are in charge
 * of calculating data
 */
template <typename T>
static int testDataParallel(int *devs, mcclComm_t *comms, int ranks, int nstreams, int blockSize,
                            int threadSizePerBlock, mcclDataType_t type)
{
    int ret = 0;
    T **sendbuff;
    T **recvbuff;
    T **srcData;
    T **t;

    T *parameter;
    T **parameter_recv;
    T *d_out_recv;
    T **d_out;

    mcStream_t *s;
    mcStream_t **s_kernel;

    int size = blockSize * threadSizePerBlock;
    /*init device buffer, stream and host buffer, split data to others*/
    initBuff<T>(devs, sendbuff, recvbuff, t, s, ranks, size);

    {
        mcSetDevice(devs[0]);
        srcData = (T **)malloc(ranks * sizeof(T *));
        for (int i = 0; i < ranks; ++i) {
            mcMalloc((void **)(srcData + i), size * sizeof(T));
        }
        /*init data*/
        for (int i = 0; i < ranks; ++i) {
            T tmp = T(i + 1);
            for (int j = 0; j < size; ++j) {
                t[i][j] = tmp;
            }
            mcMemcpy(srcData[i], t[i], size * sizeof(T), mcMemcpyHostToDevice);
        }
    }

    /* device root to all devices*/
    mcclGroupStart();
    for (int i = 0; i < ranks; i++) {
        mcclSend(srcData[i], size, type, i, comms[0], s[0]);
        mcclRecv(recvbuff[i], size, type, 0, comms[i], s[i]);
    }
    mcclGroupEnd();

    /* synchronizing streams */
    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        mcStreamSynchronize(s[i]);
    }

    /* forward compute, lunch a compute kernel with devices and streams */
    d_out = (T **)malloc(ranks * sizeof(T *));

    s_kernel = (mcStream_t **)malloc(ranks * sizeof(mcStream_t *));

    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        s_kernel[i] = (mcStream_t *)malloc(nstreams * sizeof(mcStream_t));

        mcMalloc((void **)&d_out[i], sizeof(T));
        mcMemset(d_out[i], 0, sizeof(T));

        int blockSize  = size / (nstreams * threadSizePerBlock);
        int threadSize = threadSizePerBlock;

        for (int j = 0; j < nstreams; ++j) {
            mcStreamCreate(s_kernel[i] + j);
            data_parallel<T><<<blockSize, threadSize, 0, s_kernel[i][j]>>>(
                recvbuff[i] + j * (size / nstreams), d_out[i], size / nstreams);
            mcStreamSynchronize(s_kernel[i][j]);
        }
    }

    /*gather results*/
    do {
        mcSetDevice(devs[0]);
        mcMalloc((void **)&d_out_recv, ranks * sizeof(T));
        mcMemset(d_out_recv, 0, ranks * sizeof(T));

        /* all devices to root device */
        mcclGroupStart();
        for (int i = 0; i < ranks; i++) {
            mcclSend(d_out[i], 1, type, 0, comms[i], s[i]);
            mcclRecv(d_out_recv + i, 1, type, i, comms[0], s[0]);
        }
        mcclGroupEnd();

        /* synchronizing streams */
        for (int i = 0; i < ranks; ++i) {
            mcSetDevice(devs[i]);
            mcStreamSynchronize(s[i]);
        }
    } while (0);

    /*update parameter */

    do {
        mcSetDevice(devs[0]);
        mcMalloc((void **)&parameter, sizeof(T));
        mcMemset(parameter, 0, sizeof(T));

        data_parallel_B<T><<<1, ranks, 0, s[0]>>>(d_out_recv, parameter);
        mcStreamSynchronize(s[0]);
    } while (0);

    /* broadcast to others*/
    parameter_recv = (T **)malloc(ranks * sizeof(T *));
    mcclGroupStart();
    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        mcMalloc((void **)(parameter_recv + i), sizeof(T));
        mcMemset(parameter_recv[i], 0, sizeof(T));
        mcclBroadcast((const void *)parameter, (void *)parameter_recv[i], 1, type, 0, comms[i],
                      s[i]);
    }
    mcclGroupEnd();
    /* synchronizing streams to wait for completion of mccl operation */
    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        mcStreamSynchronize(s[i]);
    }

    T check = 0;
    for (int i = 1; i <= ranks; ++i) {
        check += i * size;
    }

    T **parameter_recv_h = (T **)malloc(ranks * sizeof(T *));
    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        parameter_recv_h[i] = (T *)malloc(sizeof(T));
        memset(parameter_recv_h[i], 0, sizeof(T));
        mcMemcpy(parameter_recv_h[i], parameter_recv[i], sizeof(T), mcMemcpyDeviceToHost);
        if (*parameter_recv_h[i] != check) {
            ret = 1;
        }
    }

    /*release*/
    do {
        mcSetDevice(devs[0]);
        for (int i = 0; i < ranks; ++i) {
            mcFree(srcData[i]);
        }
        free(srcData);
        mcFree(d_out_recv);
        mcFree(parameter);
    } while (0);

    for (int i = 0; i < ranks; ++i) {
        mcSetDevice(devs[i]);
        mcFree(d_out[i]);
        mcFree(parameter_recv[i]);
        for (int j = 0; j < nstreams; ++j) {
            mcStreamDestroy(s_kernel[i][j]);
        }
        free(parameter_recv_h[i]);
        free(s_kernel[i]);
    }

    free(parameter_recv_h);
    free(parameter_recv);
    free(s_kernel);
    free(d_out);

    deinitBuff<T>(devs, sendbuff, recvbuff, t, s, ranks);
    return ret;
}

// benchmark for data_parallel
template <typename T> static void BM_data_parallel()
{
    int ndevs              = 0;
    int nstreams           = NUM_STREAMS;
    int blocksPerGrid      = GRID_SIZE;
    int threadsPerBlock    = BLOCK_SIZE;
    int size               = blocksPerGrid * threadsPerBlock;
    int nranks             = 0;
    int ret                = 0;
    int num_gpus           = 0;

    mcGetDeviceCount(&ndevs);
    printf("device count:%d\n", ndevs);
    if (ndevs < 2) {
        printf("This test requires at least two GPUs!\n");
        return;
    }
    nranks = ndevs;
    mcclDataType_t inputType = matchMcclDataType(DATA_TYPE);

    printf("devices: %d, streams: %d, blocksPerGrid: %d, threadsPerBlock: %d, type: %d\n", ndevs, nstreams,
           blocksPerGrid, threadsPerBlock, inputType);

    mcclComm_t *comms = (mcclComm_t *)malloc(nranks * sizeof(mcclComm_t));
    int *devs         = (int *)malloc(nranks * sizeof(int));

    for (int i = 0; i < nranks; i++) {
        devs[i] = i;
    }

    /* init communicators */
    mcclCommInitAll(comms, nranks, devs);

    ret += testDataParallel<T>(devs, comms, nranks, nstreams, blocksPerGrid, threadsPerBlock,
                                inputType);

    /* destory communicators */
    for (int i = 0; i < nranks; ++i) {
        mcclCommDestroy(comms[i]);
    }

    free(comms);
    free(devs);

    if (0 != ret) {
        printf("%s FAILED: %d errors\n", __func__, ret);

    } else {
        printf("%s PASSED!\n", __func__);
    }

}

int main()
{
    BM_data_parallel<float>();
}
