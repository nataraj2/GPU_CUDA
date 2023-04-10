# How to offload a for-loop onto a CUDA GPU device

This repository contains a minimal working example of how to offload 
for-loops onto a CUDA GPU device.

# Explanation of the code 
Consider a simple three-dimensional for-loop as   
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
 
# Run the example in Google Colab  
The example can also be run on Google Colab. The notebook ```GPU_CUDA_Colab.ipynb``` can be run as it is on Google Colab.
