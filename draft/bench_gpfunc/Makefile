FC = gfortran
#FFLAGS = -Wall -march=native -O2 -g -fbacktrace # -ffast-math -fopenmp
FFLAGS = -Wall -O0 -mno-avx -mno-avx2 -fno-fast-math -g -fbacktrace
SOURCES = gpfunc_bench.f90 bench_inline.f90
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

bench_inline.x: $(SOURCES)
	$(FC) $(FFLAGS) $(SOURCES) -o $@

#%.o: %.f90
#	$(FC) $(FFLAGS) -c $< -o $@

clean:
	rm -f *.x *.so *.o *.mod gpfunc.pyf
