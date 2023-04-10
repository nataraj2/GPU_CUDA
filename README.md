# How to offload computation onto a CUDA GPU device?

This repository contains a minimal working example of how to offload 
computations onto a CUDA GPU device.

## Explanation of the code 
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
where ```test_function``` is a function which performs computation on ```vel``` and ```pressure```.  
The GPU implementation of this for loop in the code will look like below  
```
ParallelFor(nx, ny, nz,
	[=] DEVICE (int i, int j, int k)noexcept
	{
		test_function(i, j, k, vel, pressure);
	});
```
```ParallelFor``` is a function that takes in 4 arguments - ```nx, ny, nz``` and the function that will be offloaded 
to the GPU device. ```DEVICE``` is a macro which is defined as ```__device___``` when using GPU or expands to blank space 
when using CPU. See the header files for the definition.  
Note that the function that is to be offloaded to the device is written as a lambda function with the variables 
captured by value using the capture clause ```[=]```.


 
## Run the example in Google Colab  
The example can also be run on Google Colab. The notebook ```GPU_CUDA_Colab.ipynb``` can be run as it is on Google Colab.
