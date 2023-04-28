#include "Array4.H"

class MultiFab{

    int nx, ny, nz;
    public:

    MultiFab(int a_nx, int a_ny, int a_nz): nx(a_nx), ny(a_ny), nz(a_nz){};


    Array4<double> array()
    {
        Array4<double> *vec;
#ifdef USE_CUDA
        cudaMallocManaged((void**)&vec, sizeof(Array4<double>));
        cudaMallocManaged((void**)&(vec[0].data), nx*ny*nz*sizeof(double));
#else
		vec = (Array4<double>*)malloc(sizeof(Array4<double>));
        vec[0].data = (double*)malloc(nx*ny*nz*sizeof(double));
#endif
        vec[0].jstride = nx;
        vec[0].kstride = nx*ny;
        return vec[0];
    }
};
