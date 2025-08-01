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

#ifndef __HELPER_MACA_H_
#define __HELPER_MACA_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <helper_string.h>

/* NOTE: MACA runtime error messages */
#ifdef __INCLUDE_MC_RUNTIME_H__
static const char *__mcGetErrorName(mcError_t error)
{
    return mcGetErrorName(error);
}
#endif

/* MCFFT ERROR */
#ifdef MCFFT_H_
static const char *__mcGetErrorName(mcfftResult error)
{
    switch (error)
    {
    case MCFFT_SUCCESS:
        return "MCFFT_SUCCESS";
    case MCFFT_INVALID_PLAN:
        return "MCFFT_INVALID_PLAN";

    case MCFFT_ALLOC_FAILED:
        return "MCFFT_ALLOC_FAILED";

    case MCFFT_INVALID_TYPE:
        return "MCFFT_INVALID_TYPE";

    case MCFFT_INVALID_VALUE:
        return "MCFFT_INVALID_VALUE";

    case MCFFT_INTERNAL_ERROR:
        return "MCFFT_INTERNAL_ERROR";

    case MCFFT_EXEC_FAILED:
        return "MCFFT_EXEC_FAILED";

    case MCFFT_SETUP_FAILED:
        return "MCFFT_SETUP_FAILED";

    case MCFFT_INVALID_SIZE:
        return "MCFFT_INVALID_SIZE";

    case MCFFT_UNALIGNED_DATA:
        return "MCFFT_UNALIGNED_DATA";

    case MCFFT_INCOMPLETE_PARAMETER_LIST:
        return "MCFFT_INCOMPLETE_PARAMETER_LIST";

    case MCFFT_INVALID_DEVICE:
        return "MCFFT_INVALID_DEVICE";

    case MCFFT_PARSE_ERROR:
        return "MCFFT_PARSE_ERROR";

    case MCFFT_NO_WORKSPACE:
        return "MCFFT_NO_WORKSPACE";

    case MCFFT_NOT_IMPLEMENTED:
        return "MCFFT_NOT_IMPLEMENTED";

    case MCFFT_LICENSE_ERROR:
        return "MCFFT_LICENSE_ERROR";

    case MCFFT_NOT_SUPPORTED:
        return "MCFFT_NOT_SUPPORTED";
    }

    return "<unknown>";
}
#endif

/* MCBLAS ERROR */
#ifdef MCBLAS_MCBLAS_H_
static const char *__mcGetErrorName(mcblasStatus_t error)
{
    switch (error)
    {
    case MCBLAS_STATUS_SUCCESS:
        return "MCBLAS_STATUS_SUCCESS";

    case MCBLAS_STATUS_NOT_INITIALIZED:
        return "MCBLAS_STATUS_NOT_INITIALIZED";

    case MCBLAS_STATUS_ALLOC_FAILED:
        return "MCBLAS_STATUS_ALLOC_FAILED";

    case MCBLAS_STATUS_INVALID_VALUE:
        return "MCBLAS_STATUS_INVALID_VALUE";

    case MCBLAS_STATUS_ARCH_MISMATCH:
        return "MCBLAS_STATUS_ARCH_MISMATCH";

    case MCBLAS_STATUS_MAPPING_ERROR:
        return "MCBLAS_STATUS_MAPPING_ERROR";

    case MCBLAS_STATUS_EXECUTION_FAILED:
        return "MCBLAS_STATUS_EXECUTION_FAILED";

    case MCBLAS_STATUS_INTERNAL_ERROR:
        return "MCBLAS_STATUS_INTERNAL_ERROR";

    case MCBLAS_STATUS_NOT_SUPPORTED:
        return "MCBLAS_STATUS_NOT_SUPPORTED";

    case MCBLAS_STATUS_LICENSE_ERROR:
        return "MCBLAS_STATUS_LICENSE_ERROR";
    }

    return "<unknown>";
}
#endif

/* MCSPARSE ERROR */
#ifdef MCSPARSE_H__
static const char *__mcGetErrorName(mcsparseStatus_t error)
{
    switch (error)
    {
    case MCSPARSE_STATUS_SUCCESS:
        return "MCSPARSE_STATUS_SUCCESS";
    case MCSPARSE_STATUS_NOT_INITIALIZED:
        return "MCSPARSE_STATUS_NOT_INITIALIZED";
    case MCSPARSE_STATUS_ALLOC_FAILED:
        return "MCSPARSE_STATUS_ALLOC_FAILED";
    case MCSPARSE_STATUS_INVALID_VALUE:
        return "MCSPARSE_STATUS_INVALID_VALUE";
    case MCSPARSE_STATUS_ARCH_MISMATCH:
        return "MCSPARSE_STATUS_ARCH_MISMATCH";
    case MCSPARSE_STATUS_MAPPING_ERROR:
        return "MCSPARSE_STATUS_MAPPING_ERROR";
    case MCSPARSE_STATUS_EXECUTION_FAILED:
        return "MCSPARSE_STATUS_EXECUTION_FAILED";
    case MCSPARSE_STATUS_INTERNAL_ERROR:
        return "MCSPARSE_STATUS_INTERNAL_ERROR";
    case MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "MCSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case MCSPARSE_STATUS_ZERO_PIVOT:
        return "MCSPARSE_STATUS_ZERO_PIVOT";
    case MCSPARSE_STATUS_NOT_SUPPORTED:
        return "MCSPARSE_STATUS_NOT_SUPPORTED";
    case MCSPARSE_STATUS_INSUFFICIENT_RESOURCES:
        return "MCSPARSE_STATUS_INSUFFICIENT_RESOURCES";
    }
    return "<unknown>";
}
#endif

/* MCRAND ERROR */
#if defined(MCRAND_H_) || defined(MCRAND_KERNEL_H_)
static const char *__mcGetErrorName(mcrandStatus error)
{
    switch (error)
    {
    case MCRAND_STATUS_SUCCESS:
        return "MCRAND_STATUS_SUCCESS";
    case MCRAND_STATUS_VERSION_MISMATCH:
        return "MCRAND_STATUS_VERSION_MISMATCH";
    case MCRAND_STATUS_NOT_INITIALIZED:
        return "MCRAND_STATUS_NOT_INITIALIZED";
    case MCRAND_STATUS_ALLOCATION_FAILED:
        return "MCRAND_STATUS_ALLOCATION_FAILED";
    case MCRAND_STATUS_TYPE_ERROR:
        return "MCRAND_STATUS_TYPE_ERROR";
    case MCRAND_STATUS_OUT_OF_RANGE:
        return "MCRAND_STATUS_OUT_OF_RANGE";
    case MCRAND_STATUS_LENGTH_NOT_MULTIPLE:
        return "MCRAND_STATUS_LENGTH_NOT_MULTIPLE";
    case MCRAND_STATUS_DOUBLE_PRECISION_REQUIRED:
        return "MCRAND_STATUS_DOUBLE_PRECISION_REQUIRED";
    case MCRAND_STATUS_LAUNCH_FAILURE:
        return "MCRAND_STATUS_LAUNCH_FAILURE";
    case MCRAND_STATUS_PREEXISTING_FAILURE:
        return "MCRAND_STATUS_PREEXISTING_FAILURE";
    case MCRAND_STATUS_INITIALIZATION_FAILED:
        return "MCRAND_STATUS_INITIALIZATION_FAILED";
    case MCRAND_STATUS_ARCH_MISMATCH:
        return "MCRAND_STATUS_ARCH_MISMATCH";
    case MCRAND_STATUS_INTERNAL_ERROR:
        return "MCRAND_STATUS_INTERNAL_ERROR";
    case MCRAND_STATUS_NOT_IMPLEMENTED:
        return "MCRAND_STATUS_NOT_IMPLEMENTED";
    }
    return "<unknown>";
}
#endif

/* Output MACA error strings when a MACA host call returns an error */
template <typename T>
void __checkMacaErrors(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "MACA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), __mcGetErrorName(result), func);
        exit(1);
    }
}
#ifndef checkMacaErrors
#define checkMacaErrors(val) __checkMacaErrors(val, #val, __FILE__, __LINE__)
#endif

#define getLastMacaError(msg) __getLastMacaError(msg, __FILE__, __LINE__)
inline void __getLastMacaError(const char *errorMessage, const char *file,
                               const int line)
{
    mcError_t err = mcGetLastError();

    if (mcSuccess != err)
    {
        fprintf(stderr,
                "%s(%i) : getLastMacaError() error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                mcGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* This will only print the proper error string when calling mcGetLastError
 * but not exit program incase error detected.
 */
#define printLastMacaError(msg) __printLastMacaError(msg, __FILE__, __LINE__)

inline void __printLastMacaError(const char *errorMessage, const char *file,
                                 const int line)
{
    mcError_t err = mcGetLastError();

    if (mcSuccess != err)
    {
        fprintf(stderr,
                "%s(%i) : getLastMacaError() error :"
                " %s : (%d) %s.\n",
                file, line, errorMessage, static_cast<int>(err),
                mcGetErrorString(err));
    }
}

// General GPU Device MACA Initialization
inline int gpuDeviceInit(int devID) {
  int device_count;
  checkMacaErrors(mcGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr,
            "gpuDeviceInit() MACA error: "
            "no devices supporting MACA.\n");
    exit(EXIT_FAILURE);
  }

  if (devID < 0) {
    devID = 0;
  }

  if (devID > device_count - 1) {
    fprintf(stderr, "\n");
    fprintf(stderr, ">> %d MACA capable GPU device(s) detected. <<\n",
            device_count);
    fprintf(stderr,
            ">> gpuDeviceInit (-device=%d) is not a valid"
            " GPU device. <<\n",
            devID);
    fprintf(stderr, "\n");
    return -devID;
  }

  int computeMode = -1, major = 0, minor = 0;
  checkMacaErrors(mcDeviceGetAttribute(&computeMode, mcDeviceAttributeComputeMode, devID));
  if (computeMode == mcComputeModeProhibited) {
    fprintf(stderr,
            "Error: device is running in <Compute Mode "
            "Prohibited>, no threads can use mcSetDevice().\n");
    return -1;
  }

  checkMacaErrors(mcSetDevice(devID));
  printf("gpuDeviceInit() MACA Device [%d] \n", devID);
  return devID;
}

// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int current_device = 0, sm_per_multiproc = 0;
    int max_perf_device = 0;
    int device_count = 0;
    int devices_prohibited = 0;

    uint64_t max_compute_perf = 0;
    checkMacaErrors(mcGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() MACA error:"
                " no devices supporting MACA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best MACA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        int computeMode = -1;
        checkMacaErrors(mcDeviceGetAttribute(&computeMode, mcDeviceAttributeComputeMode, current_device));

        // If this GPU is not running on Compute Mode prohibited,
        // then we can add it to the list
        if (computeMode != mcComputeModeProhibited)
        {
            max_perf_device = current_device;
            break;
        }
        else
        {
            devices_prohibited++;
        }
        ++current_device;
    }

    if (devices_prohibited == device_count)
    {
        fprintf(stderr,
                "gpuGetMaxGflopsDeviceId() MACA error:"
                " all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    return max_perf_device;
}

inline int findMacaDevice(int argc, const char **argv)
{
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, argv, "device"))
    {
        devID = getCmdLineArgumentInt(argc, argv, "device=");

        if (devID < 0)
        {
            printf("Invalid command line parameter\n ");
            exit(EXIT_FAILURE);
        }
        else
        {
            devID = gpuDeviceInit(devID);

            if (devID < 0)
            {
                printf("exiting...\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        checkMacaErrors(mcSetDevice(devID));
    }

    return devID;
}
/* TODO:mcDeviceGetCount is NOT READY */
inline int findIntegratedGPU(void)
{
    exit(EXIT_FAILURE);
}
/* TODO:mcDeviceGetCount is NOT READY */
inline bool checkMacaCapabilities(int major_version, int minor_version)
{
    (void)major_version;
    (void)minor_version;
    exit(EXIT_FAILURE);
}

#endif /* __HELPER_MACA_H_ */
