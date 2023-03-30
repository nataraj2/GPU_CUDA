#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

#define LAUNCH_KERNEL(MT, blocks, threads, sharedMem, ... ) \
        launch_global<MT><<<blocks, threads, sharedMem>>>(__VA_ARGS__)

template<int amrex_launch_bounds_max_threads, class L>
__launch_bounds__(amrex_launch_bounds_max_threads)
__global__ void launch_global (L f0) { f0(); }

template<class T>
struct Array4{
	T* data;
	int jstride;
	int kstride;
	public:
        __host__ __device__
	T& operator()(int i, int j, int k)const noexcept{
		return data[i + j*jstride + k*kstride];
	}
};

__host__ __device__
inline void test_function(int i, int j, int k, Array4<double> const &vec) {
	vec(i, j, k) = i+j+k;		
}

template <typename F>
__device__
auto call_f(F const &f, int i, int j, int k){
	f(i, j, k);
}

template<class L>
void ParallelFor(int nx, int ny, int nz, L &&f){
	int len_xy = nx*ny;
	int len_x = nx;
	LAUNCH_KERNEL(512, 1, 256, 0,
    	[=] __device__ () noexcept{ 	
		for(int icell=0; icell<nx*ny*nz; icell++){
			int k = icell/len_xy;
			int j = (icell - k*len_xy)/len_x;
			int i = (icell - k*len_xy - j*len_x); 
			call_f(f, i, j, k);	
		
		}
	});
}

int main(){
	
	int nx = 5, ny = 4, nz = 3;

	int len_xy = nx*ny;
	int len_x = nx;

	Array4<double> vec{new double[nx*ny*nz], len_x, len_xy} ;
	
	ParallelFor(nx, ny, nz,
	[=]__device__(int i, int j, int k)noexcept
	{
		test_function(i, j, k, vec);
	});


	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			for(int k=0;k<nz;k++){
				cout << "Vec at " << i << "," << j << "," << k << " is " << vec(i,j,k) << "\n";
			}
		}
	}

return 0;
}


