# pinnedMemory - Pinned Memory

## Description

This MACA Runtime API sample is a very basic sample that implements pinned memory copy

## Key Concepts

MACA Runtime API, pinned memory

## MACA APIs involved

mcMallocHost, mcMalloc, mcMemcpy, mcMemcpyAsync, mcDeviceSynchronize, mcFreeHost, mcFree

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
    you can modify the below parameters in cases:
    DATA_SIZE: data size need to transfer

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
$ make clean
```
### Run
 $ ./pinnedMemory

## References (for more details)

