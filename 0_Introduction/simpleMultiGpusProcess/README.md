# simpleMultiGpusProcess - a simple multi-devices multi-streams multi-processes case

## Description

This MACA Runtime API sample is a very basic sample that implements element by element vector addition in multi-processes multi-devices and multi-streams.

## Key Concepts

MACA Runtime API, multi-devices multi-streams multi-processes vector Addition

## MACA APIs involved

mcFree, mcMalloc, mcSetDevice, mcStreamCreate, mcMemcpyAsync, mcStreamSynchronize, mcStreamDestroy

## Prerequisites

Download the MACA driver and SDK, install them for your corresponding platform.

### Environment

Check if the necessary environment variables are properly set. Default installation path is shown as:
```
$ env
$ export MACA_PATH=/opt/maca
$ export PATH=${MACA_PATH}/mxgpu_llvm/bin:${MACA_PATH}/bin:${MACA_PATH}/ompi/bin:$PATH
$ export LD_LIBRARY_PATH=${MACA_PATH}/lib:${MACA_PATH}/ompi/lib:$LD_LIBRARY_PATH
$ export MACA_MPS_MODE = 1 (recommended for multi-process and VF scenarios)
```

## parameters
    you can modify the below parameters in cases:
    streams_per_device: how much streams run in one devices
    run_streams: total streams in one process
    num_blocks: block number
    num_threads_per_block: thread numbers per block
    loop_times: kernel loop times

## Build and Run

### Linux
The Linux samples are built using makefiles. To use the makefiles, change the current directory to the sample directory you wish to build, and run make:
```
$ cd <sample_dir>
$ make clean
$ make
```
### Run
 ## single process
 $ ./simpleMultiGpusProcess
 ## multi process
 $ mpirun -np n ./simpleMultiGpusProcess  (n is process numbers)

## References (for more details)

