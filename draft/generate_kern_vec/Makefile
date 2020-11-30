FC = gfortran
FFLAGS = -Wall -march=native -O3 -fopenmp -g -fbacktrace
SOURCES = kernels_base.f90 gpfunc_bench.f90 bench_inline.f90
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

bench_inline.x: $(OBJECTS)
	$(FC) $(FFLAGS) $(OBJECTS) -o $@

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f *.x *.so *.o *.mod gpfunc.pyf
