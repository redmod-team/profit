program bench_inline
use gpfunc, only: build_K, build_K_sqexp, build_K_vec, kern_sqexp_vec, &
    kern_sqexp_elem, build_K_sqexp_T, build_K_vec_T, kern_sqexp_vec_T, &
    build_K_der, Kernel, CompositeKernel, kern_sqexp_1D, build_K_prod
implicit none

integer, parameter :: np = 1024
integer, parameter :: ndim = 10

real(8), allocatable :: x(:, :), xT(:,:)
real(8), allocatable :: K(:, :)
integer :: kn, kd
integer(8) :: count, count_rate, count_max
real(8) :: tic, toc

type(Kernel) :: kern_holder
type(CompositeKernel) :: prod_kern

integer(8) :: kk

! real(8), external :: kern_sqexp

allocate(K(np, np), x(np, ndim), xT(ndim, np))

do kn = 1, np
    do kd = 1, ndim
        call random_number(x(kn, kd))
    end do
end do

xT = transpose(x)

!-------------------------------------------------------------------------------
! Element-wise kernel
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K(np, ndim, x, x, K, kern_sqexp_elem)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''

!-------------------------------------------------------------------------------
! Row-wise kernel
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K_vec(np, ndim, x, x, K, kern_sqexp_vec)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_vec:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''

!-------------------------------------------------------------------------------
! Inline in loop
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K_sqexp(np, ndim, x, x, K)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_sqexp:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''


!-------------------------------------------------------------------------------
! Inline in loop array of structs
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K_sqexp_T(np, ndim, xT, xT, K)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_sqexp_T:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''


!-------------------------------------------------------------------------------
! Row-wise array of structs
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K_vec_T(np, ndim, xT, xT, K, kern_sqexp_vec_T)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_vec_T:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''



!-------------------------------------------------------------------------------
! Row-wise kernel with derived type and function pointer
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

kern_holder%kern => kern_sqexp_vec

call build_K_der(np, ndim, x, x, K, kern_holder)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_vec_der:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''


!-------------------------------------------------------------------------------
! Product kernel with derived type and function pointer
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

allocate(prod_kern%kernels(ndim))
do kk = 1, ndim
    prod_kern%kernels(kk)%kern => kern_sqexp_1D
end do

call build_K_prod(np, ndim, x, x, K, prod_kern)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_prod:', toc - tic, 'ms'
print *, K(5,5), K(np,np-5)
print *, ''

deallocate(prod_kern%kernels)
deallocate(K, x)


end program bench_inline
