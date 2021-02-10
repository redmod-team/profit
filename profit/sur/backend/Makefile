FC = gfortran
FFLAGS = -Wall -march=native -O2 -fopenmp -g -fbacktrace
PYTHON = python3
SOURCES = gpfunc.f90  #
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

all: gpfunc.pyf $(OBJECTS) kernels_base.f90
	# Can check array copies by adding -DF2PY_REPORT_ON_ARRAY_COPY=1
	f2py -m gpfunc -c $^ --f90flags='$(FFLAGS)' -lgomp
	f2py -m kernels -c kernels.f90 kernels_base.f90 --f90flags='$(FFLAGS)' -lgomp

%.o: %.f90
	$(FC) $(FFLAGS) -c $< -o $@

gpfunc.pyf: $(SOURCES)
	$(PYTHON) -m numpy.f2py -m gpfunc $^ -h gpfunc.pyf --overwrite-signature

kernels_base.f90: init_kernels.py
	$(PYTHON) init_kernels.py

clean:
	rm -f *.x *.so *.o *.mod *.pyf kernels_base.f90
