#include "MultiFab.H"

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

