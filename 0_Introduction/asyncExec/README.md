# asyncExec - Asynchronous Execution

## Description

This MACA Runtime API sample is a very basic sample to test the asynchronous execution feature.

## Key Concepts

MACA Runtime API, Asynchronous Execution

## MACA APIs involved

mcMallocHost, mcFreeHost, mcMallocAsync, mcFreeAsync, mcMemcpyAsync, mcStreamSynchronize

## Prerequisites

Download the MACA driver and SDK, install them for your corresponding platform.

### Environment

Check if the necessary environment variables are properly set. Default installation path is shown as:
```
$ env
$ export MACA_PATH=/opt/maca
$ export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:$PATH
$ export LD_LIBRARY_PATH=${MACA_PATH}/lib
```

## Parameters
you can modify the below parameters in cases:
   - NUM_ELEMENTS: number of elements
   - BLOCK_SIZE: number of threads per block
   - ITERATIONS: number of iterations

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:

#### Build
```
$ cd <sample_dir>
$ make clean
$ make
```

#### Run
```
$ cd <sample_dir>
$ make run
```

## References (for more details)

