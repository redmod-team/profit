module gpfunc
implicit none

contains

subroutine nu_L2(nd, na, xa, xb, l, nu)
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: nu(na)             ! Output: 1/2*|x|^2

  integer :: ka

  !$omp simd
  do ka = 1, na
    nu(ka) = sum(((xa(:, ka) - xb)/l)**2)/2d0
  end do
end subroutine nu_L2


subroutine d_nu_L2_dx(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. x of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = (xa(:, ka) - xb)/l**2
  end do
end subroutine d_nu_L2_dx


subroutine d_nu_L2_dl(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. l of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = -(xa(:, ka) - xb)**2/l**3
  end do
end subroutine d_nu_L2_dl


subroutine d2_nu_L2_dx2(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. l of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = 1d0/l
  end do
end subroutine d2_nu_L2_dx2


subroutine d2_nu_L2_dxdl(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. l of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = -2d0*(xa(:, ka) - xb)/l**3
  end do
end subroutine d2_nu_L2_dxdl


subroutine d2_nu_L2_dl2(nd, na, xa, xb, l, out)
  ! Gradient w.r.t. l of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = 3d0*(xa(:, ka) - xb)**2/l**4
  end do
end subroutine d2_nu_L2_dl2


subroutine kern_sqexp(nx, nu, out)
  integer, intent(in) :: nx
  real(8), intent(in) :: nu(nx)
  real(8), intent(out) :: out(nx)

  out = exp(-nu)
end subroutine kern_sqexp


subroutine build_K(nd, nxa, nxb, xa, xb, l, K, kern)
  ! Build a kernel matrix using a function `kern` to construct columns/rows
  integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
  real(8), intent(in)    :: l(nd)         ! Length scales
  real(8), intent(inout) :: K(nxa, nxb)   ! Output: kernel matrix
  external :: kern  ! Kernel function `kern(nx, nu, out)`

  integer :: kb
  real(8) :: nu(nxa)

  !$omp parallel do private(nu)
  do kb = 1, nxb
    call nu_L2(nd, nxa, xa, xb(:, kb), l, nu)
    call kern(nxa, nu, K(:, kb))
  end do
  !$omp end parallel do

end subroutine build_K


subroutine build_K_sqexp(nd, nxa, nxb, xa, xb, l, K)
  ! Build a kernel matrix using a function `kern` to construct columns/rows
  integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
  real(8), intent(in)    :: l(nd)         ! Length scales
  real(8), intent(inout) :: K(nxa, nxb)   ! Output: kernel matrix

  call build_K(nd, nxa, nxb, xa, xb, l, K, kern_sqexp)

end subroutine build_K_sqexp

end module gpfunc
