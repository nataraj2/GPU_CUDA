#include <iostream>

#include <iostream>
#include <cmath>
#include <vector>

#ifdef USE_CUDA
	#include "ParallelForGPU.H"
	#define RUN_MODE "Using GPU"
	#define SYNC cudaDeviceSynchronize()
	#define print_gpu_details print_gpu_details()
#else 
	#include "ParallelForCPU.H"
	#define USE_CUDA false
	#define RUN_MODE "Using CPU"
	#define SYNC
	#define print_gpu_details 
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

	cout << "Run mode is " << RUN_MODE << "\n";

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
	
#ifdef USE_CUDA
	print_gpu_details;	
#endif

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
