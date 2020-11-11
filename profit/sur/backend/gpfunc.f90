module gpfunc

implicit none

contains

! Build kernel matrix
subroutine build_K(xa, xb, hyp, K, kern)
  real(8), intent(in)    :: xa(:, :), xb(:, :), hyp(:)  ! Points and hyperparams
  real(8), intent(inout) :: K(:, :)  ! Result matrix
  external :: kern  ! Kernel function
  real(8) :: kern   ! Have to split these in two lines for f2py

  integer :: ka, kb, nhyp

  nhyp = size(hyp)

  do kb = 1, size(xb)
    do ka = 1, size(xa)
      K(ka, kb) = hyp(nhyp)*kern(xa(:, ka), xb(:, kb), hyp)
    end do
  end do

end subroutine build_K

function kern_sqexp(xa, xb, hyp)
  real(8) :: kern_sqexp
  real(8), intent(in) :: xa(:), xb(:), hyp(:)
  kern_sqexp = exp(-0.5d0*(sum(xb-xa)/hyp(1))**2)
end function kern_sqexp

subroutine build_K_sqexp(np, ndim, nhyp, xa, xb, hyp, K)
  integer, intent(in)    :: np, ndim, nhyp
  real(8), intent(in)    :: xa(ndim, np)
  real(8), intent(in)    :: xb(ndim, np)
  real(8), intent(in)    :: hyp(nhyp)     ! Points and hyperparams
  real(8), intent(inout) :: K(np, np)     ! Result matrix

  call build_K(xa, xb, hyp, K, kern_sqexp)
end subroutine build_K_sqexp

end module gpfunc
