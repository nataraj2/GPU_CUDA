#include "MultiFab.H"

#define GPU_MAX_THREADS 512

#define LAUNCH_KERNEL(MT, blocks, threads, sharedMem, ... ) \
        launch_global<MT><<<blocks, threads, sharedMem>>>(__VA_ARGS__)

void print_gpu_details(){
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("Device name: %s\n", prop.name);
        printf("Total global memory: %lu bytes\n", prop.totalGlobalMem);
        printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("Maximum threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Clock rate: %d kHz\n", prop.clockRate);

        int device;
        cudaGetDevice(&device);

        int mp_count;
        cudaDeviceGetAttribute(&mp_count, cudaDevAttrMultiProcessorCount, device);

        int max_threads_per_mp;
        cudaDeviceGetAttribute(&max_threads_per_mp, cudaDevAttrMaxThreadsPerMultiProcessor, device);

        int total_threads = mp_count * max_threads_per_mp;
        printf("Total number of threads on device %d: %d %d %d\n", device, mp_count, max_threads_per_mp, total_threads);
}



template<int launch_bounds_max_threads, class L>
__launch_bounds__(launch_bounds_max_threads)
__global__ void launch_global (L f0) { f0(); }

template <typename F>
DEVICE
auto call_f(F const &f, int i, int j, int k){
	f(i, j, k);
}

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


