#ifdef USE_CUDA
	#include <cuda_runtime.h>
    #include <cuda_runtime_api.h>
	#define HOST __host__
	#define DEVICE __device__
#else
	#define HOST 
	#define DEVICE 
#endif
