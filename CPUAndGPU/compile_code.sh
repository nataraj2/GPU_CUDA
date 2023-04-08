ml use -a /nopt/nrel/apps/modules/test/modulefiles
ml gcc/8.4.0 cuda/10.2.89 openmpi/4.0.4/gcc+cuda
#nvcc -x cu --expt-extended-lambda --expt-relaxed-constexpr --expt-relaxed-constexpr ParallelFor.cpp -o out
make
