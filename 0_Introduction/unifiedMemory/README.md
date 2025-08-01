# unifiedMemory

## Description

This is a very basic vectorAdd sample, which shows how to use MACA Runtime API provided by the MACA unified memory feature.

## Key Concepts

MACA Runtime API, unifiedMemory sample

## MACA APIs involved

mcFree, mcMallocManaged, mcGetLastError, mcDeviceSynchronize, mcGetErrorString

## Prerequisites

Download the MACA driver and SDK, install them for your corresponding platform.

### Environment

Check if the necessary environment variables are properly set. Default installation path is shown as:
```
$ env
$ export MACA_PATH=/opt/maca
$ export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:$PATH
$ export LD_LIBRARY_PATH=${MACA_PATH}/lib:$LD_LIBRARY_PATH
```
## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make
$ make run
$ make clean
```

## References (for more details)

