module gpfunc
implicit none

contains

pure subroutine nu_L2(nd, na, xa, xb, th, nu)
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: th(nd)             ! Inverse length scales squared
  real(8), intent(out)   :: nu(na)             ! Output: 1/2*|x|^2

  integer :: ka

  !$omp simd
  do ka = 1, na
    nu(ka) = sum(th*(xa(:, ka) - xb)**2)
  end do
end subroutine nu_L2


subroutine d_nu_L2_dx(nd, na, xa, xb, l2inv, out)
  ! Gradient w.r.t. x of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l2inv(nd)          ! Inverse length scales squared
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = (xa(:, ka) - xb)*l2inv
  end do
end subroutine d_nu_L2_dx


subroutine d2_nu_L2_dx2(nd, na, xa, xb, l2inv, out)
  ! Gradient w.r.t. l of nu_L2
  integer, intent(in)    :: nd, na             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, na), xb(nd) ! Points
  real(8), intent(in)    :: l2inv(nd)          ! Inverse length scales squared
  real(8), intent(out)   :: out(nd, na)        ! Output

  integer :: ka

  !$omp simd
  do ka = 1, na
    out(:, ka) = l2inv
  end do
end subroutine d2_nu_L2_dx2


pure subroutine kern_sqexp(nx, nu, out)
  integer, intent(in) :: nx
  real(8), intent(in) :: nu(nx)
  real(8), intent(out) :: out(nx)

  out = exp(-0.5d0*nu)
end subroutine kern_sqexp


subroutine build_K(nd, nxa, nxb, xa, xb, l2inv, K, kern)
  ! Build a kernel matrix using a function `kern` to construct columns/rows
  integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
  real(8), intent(in)    :: l2inv(nd)     ! Inverse length scales squared
  real(8), intent(inout) :: K(nxa, nxb)   ! Output: kernel matrix
  external :: kern  ! Kernel function `kern(nx, nu, out)`

  integer :: kb
  real(8) :: nu(nxa)

  !$omp parallel do private(nu)
  do kb = 1, nxb
    ! Variant which constructs only lower diagonal
    !call nu_L2(nd, nxa-kb+1, xa(:, kb:), xb(:, kb), l2inv, nu)
    !call kern(nxa-kb+1, nu(:kb), K(kb:, kb))
     call nu_L2(nd, nxa, xa, xb(:, kb), l2inv, nu)
     call kern(nxa, nu, K(:, kb))
  end do
  !$omp end parallel do

end subroutine build_K


subroutine build_K_sqexp(nd, nxa, nxb, xa, xb, l2inv, K)
  ! Build a kernel matrix for square exp. kernel
  integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
  real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
  real(8), intent(in)    :: l2inv(nd)     ! Inverse length scales squared
  real(8), intent(inout) :: K(nxa, nxb)   ! Output: kernel matrix
  external :: kern  ! Kernel function `kern(nx, nu, out)`

  integer :: kb
  real(8) :: nu(nxa)

  !$omp parallel do private(nu)
  do kb = 1, nxb
    call nu_L2(nd, nxa, xa, xb(:, kb), l2inv, nu)
    call kern_sqexp(nxa, nu, K(:, kb))
  end do
  !$omp end parallel do

end subroutine build_K_sqexp


subroutine build_dKdth_sqexp(nd, nxa, nxb, kd, xa, xb, K, dK)
  ! Build a kernel matrix for square exp. kernel
  integer, intent(in)    :: nd, nxa, nxb  ! Dimension and number of points
  integer, intent(in)    :: kd  ! Dimension towards to differentiate
  real(8), intent(in)    :: xa(nd, nxa), xb(nd, nxb) ! Points
  real(8), intent(inout) :: K(nxa, nxb)   ! Input/Output: kernel matrix
  real(8), intent(inout) :: dK(nxa, nxb)  ! Output: derivative of kernel matrix
  external :: kern  ! Kernel function `kern(nx, nu, out)`

  integer :: kb

  !$omp parallel do
  do kb = 1, nxb
    dK(:, kb) = -0.5d0*(xb(kd, kb) - xa(kd, :))**2*K(:, kb)
  end do
  !$omp end parallel do

end subroutine build_dKdth_sqexp

end module gpfunc
