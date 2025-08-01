# singleProcessMultiGpus

## Description

This MACA Runtime API sample is a very basic sample that implements element by element vector addition in single-process and multi-devices.

## Key Concepts

MACA Runtime API, single-process multi-devices vector Addition

## MACA APIs involved

mcFree, mcMalloc, mcSetDevice, mcStreamCreate, mcMemcpyAsync, mcStreamSynchronize, mcStreamDestroy

## Prerequisites

Download the MACA driver and SDK, install them for your corresponding platform.

### Environment

Check if the necessary environment variables are properly set. Default installation path is shown as:
```
$ env
$ export MACA_PATH=/opt/maca
$ export PATH=$PATH:${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin
$ export LD_LIBRARY_PATH=${MACA_PATH}/lib
```

## parameters
you can modify the below parameters in cases:
   - NUM_DEVS: number of devices used in one process
   - NUM_STREAMS: number of streams used in one process
   - GRID_SIZE: number of blocks per grid
   - BLOCK_SIZE: number of threads per block

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <dir>
$ make clean
$ make
```
### Run
 $ ./singleProcessMultiGpus
## References (for more details)

