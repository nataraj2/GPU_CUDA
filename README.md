# How to offload computation onto a CUDA GPU device?

This repository contains a minimal working example of how to offload 
computations onto a CUDA GPU device.

## Compilation and running 
The directory `CPUAndGPU` contains the code. To run the code with GPU, the machine you run needs to have a CUDA installation.
The `Makefile` has a variable `USE_CUDA` which can be defined as `true` or 
`false` and that will compile the GPU and pure CPU versions of the code respectively. `make` will compile the code.
The executable is `parallel_for_gpu.ex` or `parallel_for_cpu.ex` for the GPU and pure CPU compilations respectively. For running with GPU, 
CUDA modules should be enabled. Run with `./parallel_for_gpu.ex` or `parallel_for_cpu.ex`.

## Run in Google Colab  
The example can also be run on Google Colab. The notebook `GPU_CUDA_Colab.ipynb` is set to run as it is on Google Colab. 
To run on CPU use the `#include "ParallelForCPU.H"` and for GPU use `#include "ParallelForGPU.H"` in the main program. 
In the Google Colab page, make sure to choose `Runtime->Change runtime type->GPU` to use GPUs.

## How does it work? 
Consider a simple three-dimensional nested for-loop which performs computation within a function as below      
```
for(int i=0;i<nx;i++){
  for(int j=0;j<ny;j++){
    for(int k=0;k<nz;k++){
      test_function(i, j, k, vel, pressure);
    }
  }
}
```
where ```test_function``` is a function which performs computation on ```vel``` and ```pressure```. The GPU implementation of 
this nested for-loop will look as below. 
```
ParallelFor(nx, ny, nz,
	[=] DEVICE (int i, int j, int k)noexcept
	{
		test_function(i, j, k, vel, pressure);
	});
```
**This would be one of the major changes (there is one more that is explained in the next section) 
the user will have to make in the application code. The header files have templated functions which will offload any function written within 
the for-loop (```test_function``` in this case) onto the device i.e. the GPU, and the user does not need to know about these.**   
`ParallelFor` is a function that takes in 4 arguments - `nx, ny, nz` and the function that will be offloaded 
to the GPU device. ```DEVICE``` is a macro which is defined as `__device__` when using GPU or expands to blank space 
when using pure CPU. Similary there is a macro for `HOST` which is `__host__` or blank space depending on if we are using a GPU or CPU. 
See `GPUMacros.H` for the definitions. Note that the function that is to be offloaded to the device 
is written as a lambda function with the variables captured by value using the capture clause `[=]`. There are two 
implementations of the `ParallelFor` function - one each in the header file `ParallelForCPU.H` and `ParallelForGPU.H`, 
and a `#ifdef` in the main function `ParallelFor.cpp` is used to choose which implementation to use depending on if we are using a CPU or a GPU.  


## The need for a custom data type
Another change that will be needed in already written codes to make them GPU-enabled is discussed in this section. The variables - `vel`, `pressure`, within the function 
that needs to be offloaded to the device - `test_function`, are captured by value in `ParallelFor` using the capture clause `[=]`, and this necessitates the variables to 
have the `const` qualifier in the definition of the function as below.
```
inline void test_function(int i, int j, int k,
                          Array4<double> const &vel,
                          Array4<double> const &pressure) {
    vel(i, j, k) = i+j+k;
    pressure(i,j,k) = 2*i*j;
}
``` 
If the variables were of the standard types such as an n-dimensional array for example, and are `const`, then they cannot be modified inside the function. Hence, the trick is to create a struct `Array4`, 
for the variables, and overload the parantheses operator `()` as in `Array4.H`. This allows values of these `Array4` variables to be modified inside the function. 
Notice how the variables are accessed in a fortran style - `vel(i,j,k)`. A class `MultiFab` is defined in `MultiFab.H`, with a member function `array` that returns an 
`Array4` object on invoking. The class uses `cudaMallocManaged` or `malloc` to allocate the variables based on if we use GPU or CPU respectively. 
`cudaMallocManaged` allocates the variable in the managed (unified) memory which is accessible by both the host (CPU) and the device (GPU), hence preventing the need to create variables on the device and explicitly copying the variables from the host to the device (as opposed to `cudaMalloc` which allocates variables only on the device). This approach is an easy start to GPU-enable 
codes already written for CPU with MPI and OpenMP. The use of unified memory makes [CUDA-aware MPI](https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/) possible. One MPI rank offloading to 
one GPU is a good start to GPU-enabling already written codes.

## Explanation of the GPU kernel launch
[NVIDIA page for introduction to CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications__technical-specifications-per-compute-capability) is a very good read.  
Kernels are functions that are launched on the GPU device. CUDA kernel launches are asynchronous (i.e. non-blocking). `ParallelForGPU.H` contains the implementation of the 
templated `ParallelFor` function for GPU using CUDA. It calls a macro -  `LAUNCH_KERNEL` which launches the kernel with a specified number of threads, and the 
number of blocks being automatically determined, and with stream and shared memory (optionally). A [grid-stride loop](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) is used so that cases with the data array size
exceeding the total number of threads (which is equal to the stride inside the for-loop, equal to `numThreadsPerBlock (i.e. blockDim.x) x numThreadsPerBlock (i.e. gridDim.x)`) are automatically handled, and this results in a flexible kernel. The function launched by the kernel looks as below 
and `call_f` will call the function which does the computation inside the nested for-loops - `test_function` in this case. The templated function `ParallelFor` launches the kernel
```
template<class L>
void ParallelFor(int nx, int ny, int nz, L &&f){
        std::cout << "Launching kernel " << "\n";
        int len_xy = nx*ny;
        int len_x = nx;
        int ncells = nx*ny*nz;
        int numBlocks = (std::max(ncells,1) + GPU_MAX_THREADS - 1 )/GPU_MAX_THREADS;
        int numThreads = GPU_MAX_THREADS;
        std::cout << "Launching " << numBlocks << " blocks " << "\n";
        LAUNCH_KERNEL(GPU_MAX_THREADS, numBlocks, numThreads, 0,
            [=] DEVICE () noexcept{
            for(int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
            icell < nx*ny*nz; icell += stride){
                int k = icell/len_xy;
                int j = (icell - k*len_xy)/len_x;
                int i = (icell - k*len_xy - j*len_x);
                call_f(f, i, j, k);
            }
        });
}
```
A one-dimensional layout of the threads is used and the i, j, k indices are determined using the `oned_index (icell in the code above) = nx ny x k + j x nx + i`. `GPU_MAX_THREADS` is 
the maximum number of threads per block, and that is fixed to be 512 (The limits for CUDA 2.0+ for threads per block is 1024 and blocks per grid is 65536). In the above example, 
`GPU_MAX_THREADS` is the number of threads launched per block. 



### Determining the number of blocks
The line in the code that determines the number of blocks is 
```
int numBlocks = (std::max(ncells,1) + GPU_MAX_THREADS - 1 )/GPU_MAX_THREADS;
```
In this example, we use `GPU_MAX_THREADS=512`, and hence the number of threads per block is 512, and `ncells=nx ny nz` is the size of the data that is offloaded onto the device. 
Hence, if we have `1<=ncells<=512`, we need only 1 block. If `ncells=513`, then we need 2 blocks - 512 threads in one block and 1 thread in the next (not an efficient usage, just an example). 
This is what the above line of code does.










  


 

