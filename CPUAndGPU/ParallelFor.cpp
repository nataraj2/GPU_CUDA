#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <cmath>
#include <vector>

#ifndef USE_CUDA
#include "ParallelForGPU.H"
#define MY_VARIABLE "Using GPU"
#define SYNC cudaDeviceSynchronize()
#else 
#define MY_VARIABLE "Using CPU"
#include "ParallelForCPU.H"
#define SYNC
#endif

using namespace std;

HOST DEVICE
inline void test_function(int i, int j, int k, 
                          Array4<double> const &vel,
			  			  Array4<double> const &pressure) {
	vel(i, j, k) = i+j+k;
	pressure(i,j,k) = 2*i*j;
}

int main(){

	cout << MY_VARIABLE << "\n";

	int nx = 5, ny = 4, nz = 3;
    
	MultiFab velfab(nx, ny, nz);
	MultiFab pressurefab(nx, ny, nz);

	auto vel = velfab.array();
	auto pressure = pressurefab.array();
  
	ParallelFor(nx, ny, nz,
	[=] DEVICE (int i, int j, int k)noexcept
	{
		test_function(i, j, k, vel, pressure);
	});

	SYNC;

	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			for(int k=0;k<nz;k++){
				cout << "Vel at " << i << "," << j << "," << k << " is " << vel(i,j,k) << " " << pressure(i,j,k) << "\n";
			}
		}
	}

return 0;
}
