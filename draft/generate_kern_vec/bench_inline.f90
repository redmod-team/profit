program bench_inline
use gpfunc, only: build_K
implicit none

integer, parameter :: nx = 1024
integer, parameter :: nd = 10

real(8), allocatable :: x(:, :)
real(8), allocatable :: K(:, :)
integer :: kx, kd
integer(8) :: count, count_rate, count_max
real(8) :: tic, toc


external :: kern_sqexp

allocate(K(nx, nx), x(nd, nx))

do kx = 1, nx
    do kd = 1, nd
        call random_number(x(kd, kx))
    end do
end do

!-------------------------------------------------------------------------------
! Row-wise kernel
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K(nd, nx, nx, x, x, K, kern_sqexp)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_vec:', toc - tic, 'ms'
print *, K(5,5), K(nx,nx-5)
print *, ''


end program bench_inline
