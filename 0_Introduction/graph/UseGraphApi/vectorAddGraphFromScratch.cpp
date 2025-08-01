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
#include <stdio.h>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define WIDTH 256
#define HEIGHT 1

#define NUM (WIDTH * HEIGHT)

__global__ void vectoradd_float(const float *a, const float *b, float *c, int width, int height)
{

    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;

    int i = x;
    if (i < (width * height))
    {
        c[i] = a[i] + b[i];
    }
}

int vectoradd_with_graph_api()
{

    float *hostA;
    float *hostB;
    float *hostC;

    float *deviceA;
    float *deviceB;
    float *deviceC;

    int width = WIDTH;
    int height = HEIGHT;

    mcDeviceProp_t devProp;
    mcGetDeviceProperties(&devProp, 0);

    int i;
    int errors;

    hostA = (float *)malloc(NUM * sizeof(float));
    hostB = (float *)malloc(NUM * sizeof(float));
    hostC = (float *)malloc(NUM * sizeof(float));

    // initialize the input data
    for (i = 0; i < NUM; i++)
    {
        hostA[i] = (float)(i + 1);
        hostB[i] = (float)(i + 1) * 100.0f;
    }

    mcMalloc(&deviceA, NUM * sizeof(float));
    mcMalloc(&deviceB, NUM * sizeof(float));
    mcMalloc(&deviceC, NUM * sizeof(float));

    /********************************Create Graph*****************************/
    mcStream_t graphStream;
    mcGraph_t graph;
    std::vector<mcGraphNode_t> nodeDependencies;
    mcGraphNode_t memcpyNode1, memcpyNode2, kernelNode, memcpyNode3, memsetNodeA, memsetNodeB,
        memsetNodeC;
    mcKernelNodeParams kernelNodeParams = {0};
    mcMemsetParams memsetParams = {0};

    mcStreamCreate(&graphStream);
    mcGraphCreate(&graph, 0);

    mcGraphAddMemcpyNode1D(&memcpyNode1, graph, nullptr, 0, deviceA, hostA,
                           sizeof(float) * NUM, mcMemcpyHostToDevice);
    mcGraphAddMemcpyNode1D(&memcpyNode2, graph, nullptr, 0, deviceB, hostB,
                           sizeof(float) * NUM, mcMemcpyHostToDevice);

    // add launch kernel operation
    kernelNodeParams.func = (void *)vectoradd_float;
    kernelNodeParams.gridDim = dim3(1, 1, 1);
    kernelNodeParams.blockDim = dim3(256, 1, 1);
    kernelNodeParams.sharedMemBytes = 0;

    // Note: this place easily meet usage error
    void *kernArgs[5] = {(void *)&deviceA, (void *)&deviceB, (void *)&deviceC, &width, &height};

    kernelNodeParams.kernelParams = (void **)kernArgs;
    kernelNodeParams.extra = NULL;
    nodeDependencies.push_back(memcpyNode1);
    nodeDependencies.push_back(memcpyNode2);
    mcGraphAddKernelNode(&kernelNode, graph, nodeDependencies.data(),
                         nodeDependencies.size(), &kernelNodeParams);
    nodeDependencies.clear();
    nodeDependencies.push_back(kernelNode);

    mcGraphAddMemcpyNode1D(&memcpyNode3, graph, nodeDependencies.data(),
                           nodeDependencies.size(), hostC, deviceC,
                           sizeof(float) * NUM, mcMemcpyDeviceToHost);

    /********************************nodes, level, edge*****************************/
    std::vector<mcGraphNode_t> nodes;
    size_t nodesNumber;
    // get graph nodes number
    mcGraphGetNodes(graph, nullptr, &nodesNumber);
    nodes.reserve(nodesNumber);
    // get graph nodes
    mcGraphGetNodes(graph, nodes.data(), &nodesNumber);
    printf("graph nodes[%zd]: ", nodesNumber);
    for (size_t i = 0; i < nodesNumber; i++)
    {
        printf("%p ", nodes[i]);
    }
    printf("\n");

    /********************************Instantiate Graph Generate Executable
     * Graph*****************************/
    mcGraphExec_t graphExec;
    mcGraphNode_t errorNode;
    char logBuffer[256] = {0};
    mcGraphInstantiate(&graphExec, graph, &errorNode, logBuffer, sizeof(logBuffer));

    /********************************Launch Executable Graph*****************************/
    mcGraphLaunch(graphExec, graphStream);

    mcStreamSynchronize(graphStream);

    /********************************Destroy Executable Graph*****************************/
    mcGraphExecDestroy(graphExec);
    /********************************Destroy Graph*****************************/
    mcGraphDestroy(graph);
    mcStreamDestroy(graphStream);
    // verify the results
    errors = 0;
    for (i = 0; i < NUM; i++)
    {
        if (hostC[i] != (hostA[i] + hostB[i]))
        {
            errors++;
            printf("Error : %d: hostC: %f, hostA+hostB: %f\n", i, hostC[i], hostA[i] + hostB[i]);
            break;
        }
    }
    if (errors != 0)
    {
        printf("FAILED: %d errors\n", errors);
    }
    else
    {
        printf("PASSED!\n");
    }

    mcFree(deviceA);
    mcFree(deviceB);
    mcFree(deviceC);

    free(hostA);
    free(hostB);
    free(hostC);

    return errors;
}

int main()
{
    vectoradd_with_graph_api();
    return 0;
}
