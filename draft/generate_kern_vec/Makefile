FC = gfortran
#FFLAGS = -Wall -march=native -Og -g -fbacktrace -lblas -llapack
FFLAGS = -Wall -Warray-temporaries -march=native -O3 -fopenmp -g -fbacktrace -lblas -llapack
#FFLAGS = -Wall -march=native -O3 -fopenmp -L/opt/local/lib -lopenblas
SOURCES = ../../profit/sur/backend/gpfunc.f90 bench_inline.f90

bench_inline.x: $(SOURCES)
	$(FC) $(FFLAGS) $^ -o $@

clean:
	rm -f *.x *.so *.o *.mod gpfunc.pyf
