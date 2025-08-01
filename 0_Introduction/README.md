# 0. Introduction

### [asyncExec]
This sample demonstrates iterative vector addition using MACA async operations. Asynchronous memory operations with mcMallocAsync/mcFreeAsync, all asynchronous tasks are synchronized via mcStreamSynchronize at completion.

### [asyncMemcpy]
This example demonstrates how to use mcMemcpyAsync for asynchronous data transfer between host (CPU) and device (GPU).

### [asyncMemset]
This sample demonstrates how to use MACA mcMemsetAsync for asynchronous device memory initialization. Concurrently filling different memory regions with data validation.

### [dynamicParallelism]
This sample demonstrates MACA dynamic parallelism. Parent kernel (dynamic_parallelism_kernel) launching a child kernel (partial_sum). Child kernel computing partial sums of an input array.

### [seGraphApi]
This example demonstrates how to optimize 1D computations (e.g., vector addition) using MACA Graph API by predefining an execution graph (mcGraph_t).

### [UseStreamCapture]
This example demonstrates how to dynamically construct an execution graph (MACA Graph) for vector addition using MACA stream capture technology. The code automatically converts asynchronous operations (memory copies and kernel launches) into graph nodes via mcStreamBeginCapture and mcStreamEndCapture.

### [pinnedMemory]
This example demonstrates basic usage of mcMallocHost and does not include performance comparisons with standard malloc.

### [profilerScope]
This example demonstrates how to use the MACA profiler(mcProfilerStart /mcProfilerStop) for kernel performance analysis. Compare kernel execution with and without profiling.

### [sharedMemory]
This example demonstrates how to optimize matrix multiplication using MACA shared memory, implementing tiled matrix computation through shared memory blocks (__shared__ float As/Bs).

### [simpleIPC]
This example demonstrates multi-GPU computation using Inter-Process Communication (IPC) with one process per GPU. It shows how to share memory and events between processes working with different GPUs.

### [simpleMultiGpusProcess]
This example demonstrates multi-GPU data parallelism using streams and MPI for distributed computation. It implements a vector addition benchmark across multiple devices with concurrent execution.

### [singleProcessMultiGpus]
This example demonstrates single-process multi-GPU computation using mccl library for GPU-to-GPU communication.

### [soma]
This example demonstrates a vector addition implementation with memory pool support, utilizing the Stream-Ordered Memory Allocator for device memory management.

### [unifiedMemory]
This sample demonstrates the MACA unified memory management through mcMallocManaged. The implementation showcases a unified memory space that is simultaneously accessible by both CPU and GPU.

### [vectorAdd]
This example demonstrates how to implement a simple vector addition kernel in MACA, computing the element-wise sum of two arrays (A + B = C). 

### [vectorAdd_mcrtc]
This example demonstrates a complete implementation of bitcode-module-based vector addition. The code illustrates dynamic kernel module management through mcModule functions, and flexible kernel launching via mcModuleLaunchKernel with configurable grid/block dimensions.
