module gpfunc
implicit none

type, abstract :: GP
  contains
  procedure(kern_elem), deferred, nopass :: kern
  procedure :: build_K_poly
end type GP

type, extends(GP) :: SqExpGP
  contains
  procedure, nopass :: kern => kern_sqexp_elem
end type SqExpGP

type Kernel
    procedure(abstract_kern), pointer, nopass :: kern
end type Kernel

type Kernel1D
    procedure(kern_1d), pointer, nopass :: kern
end type Kernel1D

type CompositeKernel
    type(Kernel1D), allocatable :: kernels(:)
end type CompositeKernel

abstract interface
    pure function kern_elem(ndim, xa, xb)
      integer, intent(in) :: ndim
      real(8), intent(in) :: xa(ndim), xb(ndim)
      real(8) :: kern_elem
    end function kern_elem

    subroutine abstract_kern(np, ndim, xa, xb, out)
        integer, intent(in) :: np, ndim
        real(8), intent(in) :: xa(np,ndim), xb(ndim)
        real(8), intent(out) :: out(np)
    end subroutine abstract_kern

    subroutine kern_1d(np, xa, xb, out)
        integer, intent(in) :: np
        real(8), intent(in) :: xa(np), xb
        real(8), intent(out) :: out(np)
    end subroutine kern_1d

end interface

contains

! Build kernel matrix polymorphism
subroutine build_K_poly(this, np, ndim, xa, xb, K)
  class(GP) :: this
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(ndim, np), xb(ndim, np) ! Points and hyperparams
  real(8), intent(inout) :: K(np, np)  ! Result matrix

  integer :: ka, kb

  !$omp parallel do
  do kb = 1, size(xb, 2)
    do ka = 1, size(xa, 2)
      K(ka, kb) = this%kern(ndim, xa(:, ka), xb(:, kb))
    end do
  end do

end subroutine build_K_poly

! Build kernel matrix
subroutine build_K(np, ndim, xa, xb, K, kern)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(ndim, np), xb(ndim, np) ! Points and hyperparams
  real(8), intent(inout) :: K(np, np)  ! Result matrix
  external :: kern  ! Kernel function
  real(8) :: kern   ! Have to split these in two lines for f2py

  integer :: ka, kb

  !$omp parallel do
  do kb = 1, size(xb, 2)
    do ka = 1, size(xa, 2)
      K(ka, kb) = kern(ndim, xa(:, ka), xb(:, kb))
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
      K(ka, kb) = exp(-0.5d0*sum((xa(ka, :) - xb(kb, :))**2))
    end do
  end do
  !$omp end parallel do
end subroutine build_K_sqexp


subroutine build_K_sqexp_T(np, ndim, xa, xb, K)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(ndim, np)
  real(8), intent(in)    :: xb(ndim, np)
  real(8), intent(inout) :: K(np, np)     ! Result matrix

  integer :: ka, kb

  !$omp parallel do
  do kb = 1, np
    !$omd simd
    do ka = 1, np
      K(ka, kb) = exp(-sum((xa(:, ka) - xb(:, kb))**2/2d0))
    end do
  end do
  !$omp end parallel do
end subroutine build_K_sqexp_T


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

    kern_sqexp_elem = exp(-0.5d0*sum((xa - xb)**2))

end function kern_sqexp_elem


subroutine kern_sqexp_vec(np, ndim, xa, xb, out)
    integer, intent(in) :: np, ndim
    real(8), intent(in) :: xa(np,ndim), xb(ndim)
    real(8), intent(out) :: out(np)

    integer :: ka

    !$omp simd
    do ka = 1, np
        out(ka) = exp(-sum((xa(ka, :) - xb)**2/2d0))
    end do

end subroutine kern_sqexp_vec


subroutine kern_sqexp_vec_T(np, ndim, xa, xb, out)
  integer, intent(in) :: np, ndim
  real(8), intent(in) :: xa(ndim, np), xb(ndim)
  real(8), intent(out) :: out(np)

  integer :: ka

  !$omp simd
  do ka = 1, np
      out(ka) = exp(-sum((xa(:, ka) - xb)**2/2d0))
  end do

end subroutine kern_sqexp_vec_T


! Build kernel matrix
subroutine build_K_vec(np, ndim, xa, xb, K, kern)
    integer, intent(in)    :: np, ndim
    real(8), intent(in)    :: xa(np, ndim), xb(np, ndim) ! Points and hyperparams
    real(8), intent(inout) :: K(np, np)  ! Result matrix
    external :: kern  ! Kernel function

    integer :: kb

    !$omp parallel do
    do kb = 1, np
        call kern(np, ndim, xa, xb(kb, :), K(:, kb))
    end do
    !$omp end parallel do

  end subroutine build_K_vec


! Build kernel matrix
subroutine build_K_vec_T(np, ndim, xa, xb, K, kern)
    integer, intent(in)    :: np, ndim
    real(8), intent(in)    :: xa(ndim, np), xb(ndim, np) ! Points and hyperparams
    real(8), intent(inout) :: K(np, np)  ! Result matrix
    external :: kern  ! Kernel function

    integer :: kb

    !$omp parallel do
    do kb = 1, np
        call kern(np, ndim, xa, xb(:, kb), K(:, kb))
    end do
    !$omp end parallel do

end subroutine build_K_vec_T


!-------------------------------------------------------------------------------
! Row-wise kernel with derived type and function pointer
subroutine build_K_der(np, ndim, xa, xb, K, kern)
    integer, intent(in)    :: np, ndim
    real(8), intent(in)    :: xa(np, ndim), xb(np, ndim) ! Points and hyperparams
    real(8), intent(inout) :: K(np, np)  ! Result matrix
    type(Kernel) :: kern  ! Kernel function container

    integer :: kb

    !$omp parallel do
    do kb = 1, np
        call kern%kern(np, ndim, xa, xb(kb, :), K(:, kb))
    end do
    !$omp end parallel do

end subroutine build_K_der



subroutine kern_sqexp_1D(np, xa, xb, out)
    integer, intent(in) :: np
    real(8), intent(in) :: xa(np), xb
    real(8), intent(out) :: out(np)

    out = exp(-(xa - xb)**2/2d0)

end subroutine kern_sqexp_1D


subroutine kern_one_1D(np, xa, xb, out)
  integer, intent(in) :: np
  real(8), intent(in) :: xa(np), xb
  real(8), intent(out) :: out(np)

  out = 1d0

end subroutine kern_one_1D


!-------------------------------------------------------------------------------
! Column-wise kernel with derived type and function pointer
subroutine build_K_prod(np, ndim, xa, xb, K, kerns)
    integer, intent(in)    :: np, ndim
    real(8), intent(in)    :: xa(np, ndim), xb(np, ndim) ! Points and hyperparams
    real(8), intent(inout) :: K(np, np)  ! Result matrix
    type(CompositeKernel) :: kerns  ! Kernel function container

    real(8) :: col(np)
    integer :: kb, kd

    K = 1d0

    !$omp parallel do private(col)
    do kd = 1, ndim
        do kb = 1, np
            call kerns%kernels(kd)%kern(np, xa(:, kd), xb(kb, kd), col)
            K(:, kb) = K(:, kb)*col
        end do
    end do
    !$omp end parallel do

end subroutine build_K_prod


subroutine kern_sqexp_vec2(nd, nx, x0, x1, out)
implicit none
INTEGER*4, intent(in) :: nd
INTEGER*4, intent(in) :: nx
REAL*8, intent(in), dimension(1:nd, 1:nx) :: x0
REAL*8, intent(in), dimension(1:nd, 1:nx) :: x1
REAL*8, intent(out), dimension(1:nd, 1:nx) :: out
INTEGER*4 :: kd
INTEGER*4 :: kx
do kx = 1, nx
   do kd = 1, nd
      out(kd, kx) = exp(-0.5d0*(-x0(kd, kx) + x1(kd, kx))**2)
   end do
end do
end subroutine


! Build kernel matrix
subroutine build_K_interface(np, ndim, xa, xb, K, kern)
  integer, intent(in)    :: np, ndim
  real(8), intent(in)    :: xa(ndim, np), xb(ndim, np) ! Points and hyperparams
  real(8), intent(inout) :: K(np, np)  ! Result matrix

  interface
    pure function kern(ndim, xa, xb)
      integer, intent(in) :: ndim
      real(8), intent(in) :: xa(ndim), xb(ndim)
      real(8) :: kern
    end function kern
  end interface

  integer :: ka, kb

  !$omp parallel do
  do kb = 1, size(xb, 2)
    do ka = 1, size(xa, 2)
      K(ka, kb) = kern(ndim, xa(:, ka), xb(:, kb))
    end do
  end do

end subroutine build_K_interface

end module gpfunc
