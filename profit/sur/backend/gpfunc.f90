module gpfunc
implicit none

contains

subroutine xdiff2_L2(nd, na, xa, xb, l, xdiff2)
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: xdiff2(na)         ! Output: squared distance

  integer :: ka

  !$omp simd
  do ka = 1, na
    xdiff2(ka) = sum(((xa(:, ka) - xb)/l)**2)
  end do
end subroutine xdiff2_L2


subroutine d_xdiff2_L2_dx(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. x of xdiff2_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = 2d0*(xa(:, ka) - xb)/l**2
  end do
end subroutine d_xdiff2_L2_dx


subroutine d_xdiff2_L2_dl(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. l of xdiff2_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = -2d0*(xa(:, ka) - xb)**2/l**3
  end do
end subroutine d_xdiff2_L2_dl


subroutine kern_sqexp(nx, xdiff2, out)
  integer, intent(in) :: nx
  real(8), intent(in) :: xdiff2(nx)
  real(8), intent(out) :: out(nx)

  out = exp(-0.5d0*xdiff2)
end subroutine kern_sqexp


subroutine build_K(nd, nxa, nxb, xa, xb, l, K, kern)
  ! Build a kernel matrix using a function `kern` to construct columns/rows
  integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
  real(8), intent(in)    :: l(nd)         ! Length scales
  real(8), intent(inout) :: K(nxa, nxb)   ! Output: kernel matrix
  external :: kern  ! Kernel function `kern(nx, xdiff2, out)`

  integer :: kb
  real(8) :: xdiff2(nxa)

  !$omp parallel do private(xdiff2)
  do kb = 1, nxb
    call xdiff2_L2(nd, nxa, xa, xb(:, kb), l, xdiff2)
    call kern(nxa, xdiff2, K(:, kb))
  end do
  !$omp end parallel do

end subroutine build_K


end module gpfunc
