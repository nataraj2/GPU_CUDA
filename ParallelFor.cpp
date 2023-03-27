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
	public:
	T& operator()(int i)const noexcept{
		return data[i];
	}
};

inline void test_function(int i, Array4<double> const &vec) {
	vec(i) = i+1;		
}

template <typename F>
auto call_f(F const &f, int i){
	f(i);
}

template<class L>
void ParallelFor(int n, L &&f){
	for(int i=0;i<n;i++){
		call_f(f, i);	
	}
}

int main(){
	
	int n = 100;

	Array4<double> vec{new double[n]} ;
	
	ParallelFor(n,
	[=](int i)noexcept
	{
		test_function(i, vec);
	});

	for(int i=0;i<n;i++){
		cout << "Vec at " << i << " is " << vec(i) << "\n";
	}

return 0;
}
