FC = gfortran
FFLAGS = -Wall -march=native -O3 -fopenmp -g -fbacktrace
PYTHON = python3
SOURCES = gpfunc.f90 kernels.f90 kernels_base.f90
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

all: gpfunc.pyf $(OBJECTS)
	f2py -m gpfunc -c $^ -DF2PY_REPORT_ON_ARRAY_COPY=1

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

gpfunc.pyf: $(SOURCES)
	$(PYTHON) -m numpy.f2py -m gpfunc $^ -h gpfunc.pyf --overwrite-signature

clean:
	rm -f *.x *.so *.o *.mod gpfunc.pyf
