module gpfunc
implicit none

contains

subroutine build_K(nd, nxa, nxb, xa, xb, K, kern, l, sig2f, sig2n, periodic)
    ! Build a kernel matrix using a function `kern` to construct columns/rows
    integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
    real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
    real(8), intent(inout) :: K(nxa, nxb)   ! Output: kernel matrix
    real(8), intent(in)    :: l(nd)         ! Length scales
    real(8), intent(in)    :: sig2f         ! Output scale covariance
    real(8), intent(in)    :: sig2n         ! Noise covariance
    logical, optional      :: periodic(nd)  ! `True` for periodic directions
    external :: kern  ! Kernel function `kern(nd, nx, xa, xb, out)`

    integer :: ka, kd
    real(8) :: xanorm(nd, nxa), xbnorm(nd, nxb)

    if (present(periodic)) then
        xbnorm = 0d0
        !$omp simd
        do kd = 1, nd
            if (periodic(kd)) then
                xanorm(kd, :) = (xa(kd, :) - xb(kd, :))/l(kd)
            else
                xanorm(kd, :) = sin(xa(kd, :) - xb(kd, :))/l(kd)
            endif
        end do
    else
        !$omp simd
        do kd = 1, nd
            xanorm(kd, :) = xa(kd, :)/l(kd)
            xbnorm(kd, :) = xb(kd, :)/l(kd)
        end do
    endif
    !$omp parallel do
    do ka = 1, nxa
        call kern(nd, nxb, xanorm(:, ka), xbnorm, K(:, ka))
        K(:, ka) = sig2f*K(:, ka)
        K(ka, ka) = K(ka, ka) + sig2n
    end do
    !$omp end parallel do

end subroutine build_K


subroutine kern_sqexp(nd, nx, xa, xb, out)
    integer, intent(in) :: nd, nx
    real(8), intent(in) :: xa(nd), xb(nd, nx)
    real(8), intent(out) :: out(nx)
    integer :: kx
    real(8) :: xdiff2
    !$omp simd
    do kx = 1, nx
        xdiff2 = sum((xa - xb(:, kx))**2)
        out(kx) = exp(-0.5d0*xdiff2)
    end do
end subroutine kern_sqexp

end module gpfunc
