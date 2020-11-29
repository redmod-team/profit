module gpfunc

implicit none

contains

! Build kernel matrix
subroutine build_K(np, ndim, xa, xb, K, kern)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(np, ndim), xb(np, ndim) ! Points and hyperparams
  real(8), intent(inout) :: K(np, np)  ! Result matrix
  external :: kern  ! Kernel function
  real(8) :: kern   ! Have to split these in two lines for f2py

  integer :: ka, kb

  !$omp parallel do
  do kb = 1, size(xb, 1)
    do ka = 1, size(xa, 1)
      K(ka, kb) = kern(ndim, xa(ka, :), xb(kb, :))
    end do
  end do

end subroutine build_K


subroutine build_K_sqexp(np, ndim, xa, xb, K)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(np, ndim)
  real(8), intent(in)    :: xb(np, ndim)
  real(8), intent(inout) :: K(np, np)     ! Result matrix

  integer :: ka, kb

  !$omp parallel do
  do kb = 1, np
    !$omd simd
    do ka = 1, np
      K(ka, kb) = exp(-sum((xa(ka, :) - xb(kb, :))**2/2d0))
    end do
  end do
  !$omp end parallel do
end subroutine build_K_sqexp


subroutine build_ldKdl_sqexp(np, ndim, xa, xb, dim, dKdl)
  ! Returns matrix of l_d*dk/dl_d for dimension d
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(np, ndim)
  real(8), intent(in)    :: xb(np, ndim)
  integer, intent(in)    :: dim            ! Dimension towards to differentiate
  real(8), intent(inout) :: dKdl(np, np)   ! Derivative matrix

  real(8) :: xdiff(np, ndim)
  integer :: ka, kb

  do kb = 1, np
    do ka = 1, np
      dKdl(ka, kb) = -xdiff(ka, dim)*exp(-sum((xa(ka, :) - xb(kb, :))**2/2d0))
    end do
  end do
end subroutine build_ldKdl_sqexp


pure function kern_sqexp_elem(ndim, xa, xb)
    integer, intent(in) :: ndim
    real(8), intent(in) :: xa(ndim), xb(ndim)
    real(8) :: kern_sqexp_elem

    kern_sqexp_elem = exp(-sum((xa - xb)**2/2d0))

end function kern_sqexp_elem


subroutine kern_sqexp_vec(np, ndim, xa, xb, out)
    integer, intent(in) :: np, ndim
    real(8), intent(in) :: xa(np,ndim), xb(ndim)
    real(8), intent(out) :: out(ndim)

    integer :: ka

    !$omp parallel do
    do ka = 1, np
        out(ka) = exp(-sum((xa(ka, :) - xb)**2/2d0))
    end do
    !$omp end parallel do

end subroutine kern_sqexp_vec


! Build kernel matrix
subroutine build_K_vec(np, ndim, xa, xb, K, kern)
    integer, intent(in)    :: np, ndim
    real(8), intent(in)    :: xa(np, ndim), xb(np, ndim) ! Points and hyperparams
    real(8), intent(inout) :: K(np, np)  ! Result matrix
    external :: kern  ! Kernel function

    integer :: kb

    !$omp simd
    do kb = 1, size(xb, 1)
        call kern(np, ndim, xa, xb(kb, :), K(:, kb))
    end do

  end subroutine build_K_vec

end module gpfunc
