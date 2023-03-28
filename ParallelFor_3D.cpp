#include <iostream>
#include <cmath>
#include <vector>

using namespace std;

/*class Box{
	public:
		int xlo, xhi;
};*/

/*struct mydouble{
	double *data;
	public:
    double& operator()()const noexcept{
		return *data;
	} 
};*/

template<class T>
struct Array4{
	T* data;
	int jstride;
	int kstride;
	public:
	T& operator()(int i, int j, int k)const noexcept{
		return data[i + j*jstride + k*kstride];
	}
};

inline void test_function(int i, int j, int k, Array4<double> const &vec) {
	vec(i, j, k) = i+j+k;		
}

template <typename F>
auto call_f(F const &f, int i, int j, int k){
	f(i, j, k);
}

template<class L>
void ParallelFor(int nx, int ny, int nz, L &&f){
	int len_xy = nx*ny;
	int len_x = nx;
	for(int icell=0; icell<nx*ny*nz; icell++){
		int k = icell/len_xy;
		int j = (icell - k*len_xy)/len_x;
		int i = (icell - k*len_xy - j*len_x); 
		call_f(f, i, j, k);	
	}
}

int main(){
	
	int nx = 5, ny = 4, nz = 3;

	int len_xy = nx*ny;
	int len_x = nx;

	Array4<double> vec{new double[nx*ny*nz], len_x, len_xy} ;
	
	ParallelFor(nx, ny, nz,
	[=](int i, int j, int k)noexcept
	{
		test_function(i, j, k, vec);
	});

	// Print out for testing: vec(i,j,k) = i + j + k

	for(int i=0;i<nx;i++){
		for(int j=0;j<ny;j++){
			for(int k=0;k<nz;k++){
				cout << "Vec at " << i << "," << j << "," << k << " is " << vec(i,j,k) << "\n";
			}
		}
	}

return 0;
}
