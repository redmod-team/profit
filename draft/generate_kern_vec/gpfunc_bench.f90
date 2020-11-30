module gpfunc
implicit none

contains

! Build kernel matrix
subroutine build_K(nd, nxa, nxb, xa, xb, K, kern)
    integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
    real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
    real(8), intent(inout) :: K(nxa, nxb)  ! Result matrix
    external :: kern  ! Kernel function

    integer :: ka

    !$omp parallel do
    do ka = 1, nxa
        call kern(nd, nxb, xa(:, ka), xb, K(:, ka))
    end do

end subroutine build_K

subroutine kern_sqexp(nd, nx, xa, xb, out)
    implicit none
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
