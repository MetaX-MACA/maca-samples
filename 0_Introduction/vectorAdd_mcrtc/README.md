# vectorAdd_mcrtc - Vector Addition with runtime compilation

## Description

This MACA Runtime API sample uses mcrtc for runtime compilation, it is a very basic sample that implements element by element vector addition.

## Key Concepts

MACA Runtime API, Vector Addition, Runtime Compilation

## MACA APIs involved

compileFileToBitcode, mcModuleGetFunction, loadCode, mcModuleLaunchKernel, mcDeviceSynchronize

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

