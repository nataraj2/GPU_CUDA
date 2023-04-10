#define HOST __host__
#define DEVICE __device__

#define LAUNCH_KERNEL(MT, blocks, threads, sharedMem, ... ) \
        launch_global<MT><<<blocks, threads, sharedMem>>>(__VA_ARGS__)

template<typename T>
struct Array4{
	T* data;
	int jstride;
	int kstride;

  constexpr Array4(T* a_p): data(a_p){};

	public:
    	HOST DEVICE
		T& operator()(int i, int j, int k)const noexcept{
						return data[i + j*jstride + k*kstride];
		}
};

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
		LAUNCH_KERNEL(512, 2, 256, 0,
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

template <typename T>
HOST DEVICE inline
Array4<T>
makeArray4 (T* p) noexcept
{ 
    return Array4<T>{p};
}

class MultiFab{
		
	int nx, ny, nz;
	public:
		
	MultiFab(int a_nx, int a_ny, int a_nz): nx(a_nx), ny(a_ny), nz(a_nz){};

	Array4<double> array()
	{
		Array4<double> *vec;
		cudaMallocManaged((void**)&vec, sizeof(Array4<double>));
		cudaMallocManaged((void**)&(vec[0].data), nx*ny*nz*sizeof(double));
		vec[0].jstride = nx;
		vec[0].kstride = nx*ny;
		return vec[0];
	}		
};
