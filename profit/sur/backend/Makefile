FC = gfortran
SOURCES = gpfunc.f90 kernels.f90 kernels_base.f90
OBJECTS = $(patsubst %.f90, %.o, $(SOURCES))

all: $(OBJECTS)

%.o: %.f90
	$(FC) -c $< -o $@

gpfunc.pyf: $(SOURCES)
	f2py $^

clean:
	rm -f *.x *.so *.o *.mod
