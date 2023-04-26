#include "GPUMacros.H"

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

template <typename T>
HOST DEVICE inline
Array4<T>
makeArray4 (T* p) noexcept
{
    return Array4<T>{p};
}
