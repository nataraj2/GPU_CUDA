#define HOST 
#define DEVICE 

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

template <typename F>
auto call_f(F const &f, int i, int j, int k){
	f(i, j, k);
}

template<class L>
void ParallelFor(int nx, int ny, int nz, L &&f){
	for(int i=0;i<nx;i++){
	  for(int j=0;j<ny;j++){
				for(int k=0;k<nz;k++){
				call_f(f, i, j, k);	
			}
    }
  }
}

class MultiFab{
		
	int nx, ny, nz;

	public:
		
	MultiFab(int a_nx, int a_ny, int a_nz): nx(a_nx), ny(a_ny), nz(a_nz){};

	Array4<double> array(){
		Array4<double> *vec;
        vec = (Array4<double>*)malloc(sizeof(Array4<double>));
        vec[0].data = (double*)malloc(nx*ny*nz*sizeof(double));
				vec[0].jstride = nx;
				vec[0].kstride = nx*ny;
				return vec[0];
	}		
};
