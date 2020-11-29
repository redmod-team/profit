program bench_inline
use gpfunc, only: build_K, build_K_sqexp, build_K_vec, kern_sqexp_vec, kern_sqexp_elem
implicit none

integer, parameter :: np = 1024
integer, parameter :: ndim = 4

real(8), allocatable :: x(:, :)
real(8), allocatable :: K(:, :)
integer :: kn, kd
integer(8) :: count, count_rate, count_max
real(8) :: tic, toc

! real(8), external :: kern_sqexp

allocate(K(np, np), x(np, ndim))

do kn = 1, np
    do kd = 1, ndim
        call random_number(x(kn, kd))
    end do
end do

!-------------------------------------------------------------------------------
! Element-wise kernel
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K(np, ndim, x, x, K, kern_sqexp_elem)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K:', toc - tic, 'ms'

!-------------------------------------------------------------------------------
! Row-wise kernel
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K_vec(np, ndim, x, x, K, kern_sqexp_vec)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_vec:', toc - tic, 'ms'

!-------------------------------------------------------------------------------
! Inline in loop
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K_sqexp(np, ndim, x, x, K)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_sqexp:', toc - tic, 'ms'



deallocate(K, x)


end program bench_inline
