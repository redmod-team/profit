Design
======

The general design guidline is in the spirit of "data-oriented" programming.
In particular this means

* Keep it simple and stupid.
* Program for the most frequent use cases.
* Think "what does the machine do with what data" rather than in abstract
  entities.
* Optimize for performance at design level. There is no "too early
  optimization".
* If there is one, there are many: Vectorize down to lowest level routines.

Other guidlines (see book "a philosophy of software design" by J. Ousterhout):

* Deep interfaces: Don't write routines if it takes more effort to call them
  than to spell out their content.


Conclusion on design of GP Regression
-------------------------------------

These points summarize a design balancing performance and flexibility
as found in the detailed study below.

1) There is only one kernel allowed for all dimensions
2) Inputs (periodic, shift/scale) and outputs (shift/scale) can be transformed
3) Ordering is "array of points" as in most Machine Learning libraries
   (in contrast to storing a contiguous array for individual coordinates)
4) Building the covariance matrix is parallelized over cores in the outer loop
5) Kernel functions yield covariance matrix rows and are SIMD-parallel

Gaussian Process Regression
---------------------------

Remark on notation:
For general ordered sequences of elements with no notion of direction
the term "tuple" is used. The term "vector" is used either for a directed
geometrical "arrow-like" object, or for a tuple of numbers for linear algebra.
In contrast, "vectorize" means processing array quantities to use SIMD and
parallel hardware (GPUs, accelerators).

GP regression is built on covariance matrices. These are constructed by
evaluation of kernel function over pairs of point tuples. In practice kernels
are often stationary, i.e. depend only on the (scalar) distance between points.
A weaker variant of stationarity is to allow the kernel to depend on the
vectorial distance.

Since a kernel is rarely applied to a pair of two individual points, it makes
sense to immediately vectorize the operation. Memory requirements of a full
$m \times n$ matrix scale accordingly, and reach 4 GB for m=n=10.000 points.
This is why we don't want to construct the matrix of all pairs explicitly, but
rather only single rows of the matrix, in order to take products with vectors.
In that case (see also GPyTorch) one can use iterative methods for matrices
larger than main memory.

Kernel matrices are always symmetric and positive definite. This means that the
appropriate direct solver is Cholesky. As an input for a direct solver only
a triangular matrix has to be stored due to mirror symmetry. An iterative solver
that relies only on matrix-vector products is CG. One can precondition CG
with incomplete or pivoted Cholesky (GPyTorch). A related option is the
decomposition using a truncated eigenspectrum from an iterative Lanczos
method. The result should be the same as in CG, as both methods rely on the
same Krylov subspace. Both don't require construction of the whole matrix,
but Lanczos provides a final decomposition at the cost of additional memory.

In case of short length scales in low dimensions, most points are decoupled and
many entries of the covariance matrix become zero. In that case, Wendland
kernels with compact support can be used to construct sparse covariance
matrices. These are again suited for direct solution and storage. However,
the inverse covariance matrix is not sparse, so not much is gained for
construction of surrogates.

Example on data oriented optimization
-------------------------------------

The code for this is found in `draft/bench_gpfunc`.

Three variants of assembling a kernel matrix were tested on n=1024 points
and tested on an i7 4770K CPU (Haswell, 16 FLOPS DP per cycle in AVX register):

1) Explicitly writing the kernel in a double loop  (18 ms)
2) Calling a kernel function for each element in the loop (115 ms)
3) Calling a vectorized kernel function for each row of the matrix (18 ms)

Variant 2 is much slower and can be made a bit faster (40 ms) by directly
acting on the vector difference between points. Still it is at least 2 times
less efficient than the fully inlined version 1. Variant 3 has the same
performance as putting the innermost loop in the row-wise kernel permits
cache and SIMD optimization.

On a quad-core CPU, case 1 and 3 can be brought down to 5-10 ms by combining
OpenMP PARALLEL and SIMD statements. Variant 2 still remains at 30-35 ms.
Here, case 1 can be about 20% faster than 3 due to better optimization of
parallel loops.

Array or struct or struct of arrays
-----------------------------------

Some practical tests have shown that array of structs vs struct of arrays
is mostly negligible, so we stick to the usual convention of using
struct of arrays.


Thoughts on GPUs (end of 2020)
------------------------------

CPU: Vector extensions support 8-32 FLOPS per clock cycle, i.e.
50-100 GFLOPS per core, or about 1 TFLOP on a modern 16-core machine.

Modern GPUs offer up to 30 TFLOPs single-precision but only 500 GFLOPs
double-precision performance in commodity devices. Testing GPyTorch in
parallel to a CPU implementation will give an idea on the performance.

Management of data
------------------

Traditionally, high-performance routines (BLAS/LAPACK) in Fortran require
the user to allocate data structures and supply them to the routine, which
reads and writes them. In more recent software this is often hidden from
the user. Here we take the traditional approach at least to the mid level
of the API and hide data only where appropriate. For example, the kernel
matrix is supplied explicitly as an intent(inout) parameter to routines.
Keeping only a single copy of this matrix is necessary due to memory
constraints.

Kernels in Fortran
------------------

Kernels in Python/NumPy, are about factor 1-3 slower than a native
implementation in Fortran (draft/fortran_vs_numpy). A short try in Julia
yielded problems with temporary array allocations which make it much slower
than either in this implementation. The problem there is that Julia creates
array copies per default, even over the contiguous axis. This is prevented
by adding the `@views` macro in front of a function call taking array
slices `x[:, i]`.
