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

  do kb = 1, size(xb, 1)
    do ka = 1, size(xa, 1)
      K(ka, kb) = hyp(nhyp)*kern(xa(ka, :) - xb(kb, :), hyp(:size(xa, 2)))
    end do
  end do

end subroutine build_K


subroutine build_K_sqexp(np, ndim, xa, xb, l, K)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(np, ndim)
  real(8), intent(in)    :: xb(np, ndim)
  real(8), intent(in)    :: l(ndim)       ! Length scales
  real(8), intent(inout) :: K(np, np)     ! Result matrix

  real(8) :: xdiff(np, ndim)
  integer :: ka, kb

  do kb = 1, np
    xdiff = (xa(:, :) - xb(kb, :)) / (2.0d0*l)
    do ka = 1, np
      K(ka, kb) = exp(-sum(xdiff(ka, :)**2))
    end do
  end do
end subroutine build_K_sqexp


subroutine build_dKdl_sqexp(np, ndim, xa, xb, l, K)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(np, ndim)
  real(8), intent(in)    :: xb(np, ndim)
  real(8), intent(in)    :: l(ndim)       ! Length scales
  real(8), intent(inout) :: K(np, np)     ! Result matrix

  real(8) :: xdiff(np, ndim)
  integer :: ka, kb

  do kb = 1, np
    xdiff = (xa(:, :) - xb(kb, :)) / (2.0d0*l)
    do ka = 1, np
      K(ka, kb) = exp(-sum(xdiff(ka, :)**2))
    end do
  end do
end subroutine build_dKdl_sqexp


end module gpfunc
