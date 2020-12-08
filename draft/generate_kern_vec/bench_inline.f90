program bench_inline
use gpfunc, only: build_K, kern_sqexp
implicit none

integer, parameter :: nx = 4096
integer, parameter :: nd = 4

real(8), allocatable :: x(:, :), y(:), l(:)
real(8), allocatable :: K(:, :)
integer :: kx, kd
integer(8) :: count, count_rate, count_max
real(8) :: tic, toc

integer :: info

external :: dpotrf
external :: dtrmv

allocate(K(nx, nx), x(nd, nx), l(nd), y(nx))

l = 1d0

do kx = 1, nx
    call random_number(y(kx))
    do kd = 1, nd
        call random_number(x(kd, kx))
    end do
end do

!-------------------------------------------------------------------------------
! Row-wise kernel
K = 0d0
call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call build_K(nd, nx, nx, x, x, l, K, kern_sqexp)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time build_K_vec:', toc - tic, 'ms'
print *, K(5,5), K(nx,nx-5)
print *, ''

! Nugget regularization
do kx = 1, nx
    K(kx, kx) = K(kx, kx) + 1d-10
end do

call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

! This doesn't touch anything above diagonal
print *, K(1,1:3)
print *, K(2,1:3)
print *, K(3,1:3)
call dpotrf('L', nx, K, nx, info)
print *, '------------------'
print *, K(1,1:3)
print *, K(2,1:3)
print *, K(3,1:3)

call system_clock(count, count_rate, count_max)

toc = count*1d3/count_rate
print *, 'Result Cholesky:', info
print *, 'Time Cholesky:', toc - tic, 'ms'
print *, ''

call system_clock(count, count_rate, count_max)
tic = count*1d3/count_rate

call dtrmv('L', 'N', 'N', nx, K, nx, y, 1)

call system_clock(count, count_rate, count_max)
toc = count*1d3/count_rate
print *, 'Time matmul:', toc - tic, 'ms'

deallocate(K, x, y, l)


end program bench_inline
