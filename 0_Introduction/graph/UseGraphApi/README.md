# UseGraphApi - Use Graph Api

## Description

This example of the MACA runtime API is a very basic example of using Graph

## Key Concepts

MACA Runtime API, Graph

## MACA APIs involved

mcGetDeviceProperties,mcMalloc,mcStreamCreate,mcGraphCreate,mcGraphAddMemcpyNode,mcGraphAddMemcpyNode1D,mcGraphAddKernelNode,mcGraphGetNodes,mcGraphInstantiate,mcGraphLaunch,mcStreamSynchronize,mcGraphExecDestroy,mcGraphDestroy,mcStreamDestroy,mcFree

## Prerequisites

Download the MACA driver and SDK, install them for your corresponding platform.

### Environment

Check if the necessary environment variables are properly set. Default installation path is shown as:
```
$ env
$ export MACA_PATH=/opt/maca
$ export PATH=$PATH:${MACA_PATH}/mxgpu_llvm/bin
$ export LD_LIBRARY_PATH=${MACA_PATH}/lib
```

## parameters

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
$ make clean
```
### Run
 $ ./vectorAddGraphFromScratch

## References (for more details)

