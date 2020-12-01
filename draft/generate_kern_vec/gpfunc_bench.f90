module gpfunc
implicit none

contains

subroutine xdiff2_L2(nd, nb, xa, xb, l, xdiff2)
  integer, intent(in)    :: nd, nb             ! Dimension and number of points
  real(8), intent(in)    :: xa(nd), xb(nd, nb) ! Points
  real(8), intent(in)    :: l(nd)              ! Length scales
  real(8), intent(out)   :: xdiff2(nb)         ! Output: squared distance

  integer :: kb

  !$omp simd
  do kb = 1, nb
    xdiff2(kb) = sum(((xa - xb(:, kb))/l)**2)
  end do
end subroutine xdiff2_L2


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

  integer :: ka !, kb, kd
  real(8) :: xdiff2(nxb) ! xanorm(nd, nxa), xbnorm(nd, nxb)

  ! if (present(periodic)) then
  !   xbnorm = 0d0
  !   !$omp simd
  !   do kd = 1, nd
  !     if (periodic(kd)) then
  !       xanorm(kd, :) = sin(xa(kd, :) - xb(kd, :))/l(kd)
  !     else
  !       xanorm(kd, :) = (xa(kd, :) - xb(kd, :))/l(kd)
  !     endif
  !   end do
  ! else
  !   !$omp simd
  !   do kd = 1, nd
  !     xanorm(kd, :) = xa(kd, :)/l(kd)
  !     xbnorm(kd, :) = xb(kd, :)/l(kd)
  !   end do
  ! endif

  !$omp parallel do private(xdiff2)
  do ka = 1, nxa
    call xdiff2_L2(nd, nxb, xa(:, ka), xb, l, xdiff2)
    call kern(nxb, xdiff2, K(:, ka))
    K(:, ka) = sig2f*K(:, ka)
    K(ka, ka) = K(ka, ka) + sig2n
  end do
  !$omp end parallel do

end subroutine build_K


! TODO: Build dK/dl(kd)
! use dkern/kern as below
!
! TODO: Build K derivative observarion
! for sqexp Kxx = (1 - x)*K or something like this
! could give instead of dkern(x) better f=dkern(x)/kern(x)
! f : dkern = f*kern  (auto-differentiation and pullback similar)
! very cheap for sqexp (and periodic)
!
! build_L
!   loop over dimension and evaluate dkern d2kern
!
! build dL/dl(kd)


subroutine kern_sqexp(nx, xdiff2, out)
  integer, intent(in) :: nx
  real(8), intent(in) :: xdiff2(nx)
  real(8), intent(out) :: out(nx)

  out = exp(-0.5d0*xdiff2)
end subroutine kern_sqexp

end module gpfunc
