# How to offload computation onto a CUDA GPU device?

This repository contains a minimal working example of how to offload 
computations onto a CUDA GPU device.

## Running 
The directory ```CPUAndGPU``` contains the code. To run the code with GPU, the machine you run needs to have a CUDA installation.
The ```Makefile``` has a variable ```USE_CUDA``` which can be defined as ```true``` or 
```false``` and that will compile the GPU and pure CPU versions of the code respectively. 
The executable is ```parallel_for_gpu.ex``` or ```parallel_for_cpu.ex``` for the GPU and pure CPU compilations respectively.

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
this nested for-loop will look as below. **This would be one of the major changes (there is one more that is explained in a later section) 
the user will have to make in the application code. The header files have templated functions which will offload any function written within 
the for-loop (```test_function``` in this case) onto the device i.e. the GPU.** 
```
ParallelFor(nx, ny, nz,
	[=] DEVICE (int i, int j, int k)noexcept
	{
		test_function(i, j, k, vel, pressure);
	});
```
```ParallelFor``` is a function that takes in 4 arguments - ```nx, ny, nz``` and the function that will be offloaded 
to the GPU device. ```DEVICE``` is a macro which is defined as ```__device__``` when using GPU or expands to blank space 
when using pure CPU. Similary there is a macro for `HOST` which is `__host__` or blank space depending on if we are using a GPU or CPU. 
See `GPUMacros.H` for the definitions. Note that the function that is to be offloaded to the device 
is written as a lambda function with the variables captured by value using the capture clause ```[=]```. There are two 
implementations of the ```ParallelFor``` function - one each in the header file ```ParallelForCPU.H``` and ```ParallelForGPU.H```, 
and a ```#ifdef``` is used to choose which implementation to use depending on if we are using a CPU or a GPU.  

## ```ParallelFor``` for GPU
```ParallelForGPU.H``` contains the implementation of the ```ParallelFor``` function for GPU using CUDA. It calls a macro -  ```LAUNCH_KERNEL``` 
which launches the kernel with the specified number of blocks, threads, stream and shared memory (optionally). 
A [grid-stride loop](https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/) is used so that cases with the data array size
exceeding the number of threads are automatically handled, and this 
results in a flexible kernel. The function launched by the kernel looks as below 
```
for(int icell = blockDim.x*blockIdx.x+threadIdx.x, stride = blockDim.x*gridDim.x;
	icell < nx*ny*nz; icell += stride){
		int k = icell/len_xy;
		int j = (icell - k*len_xy)/len_x;
		int i = (icell - k*len_xy - j*len_x); 
		call_f(f, i, j, k);	
}
```
and ```call_f``` will call the function which does the computation inside the nested for-loops - ```test_function``` in this case.

## Creating a data type
The vaariables wihin the function that needs to be offloaded to the device - `test_function`, are captured by value, and this 
necessitates the variables to have the `const` qualifier in the definition of the function. 
```
inline void test_function(int i, int j, int k,
                          Array4<double> const &vel,
                          Array4<double> const &pressure) {
    vel(i, j, k) = i+j+k;
    pressure(i,j,k) = 2*i*j;
}
``` 
But usually, if the variables are `const` then it cannot be modified. Hence, a `struct Array4` is defined and the parantheses operator 
`()` is overloaded as in `Array4.H`. This allows values of these `Array4` variables to be updated.




 
## Run the example in Google Colab  
The example can also be run on Google Colab. The notebook ```GPU_CUDA_Colab.ipynb``` can be run as it is on Google Colab. 
To run on CPU use the ```#include "ParallelForCPU.H"``` and for GPU use ```#include "ParallelForGPU.H"```.
