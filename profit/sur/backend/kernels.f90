module kernels
implicit none

contains

function kern_prod(k, x, l) result(y)
  external :: k        ! Base kernel
  real(8) :: k
  real(8), intent(in)  :: x(:)  ! Input argument
  real(8), intent(in)  :: l(:)  ! Length scales
  real(8) :: y
  integer :: i
  y = 1d0
  do i = 1, size(x)
    y = y*k(x(i)/l(i))
  end do
end function

function dkern_prod(k, dk, j, x, l) result(y)
  external :: k        ! Base kernel
  real(8) :: k
  external :: dk        ! Base kernel
  real(8) :: dk       ! Derivative of base kernel
  integer, intent(in) :: j      ! Direction to differentiate
  real(8), intent(in)  :: x(:)  ! Input argument
  real(8), intent(in)  :: l(:)  ! Length scales
  real(8) :: y
  integer :: i
  y = 1d0
  do i = 1, size(x)
    if (i == j) then
      y = y*dk(x(i)/l(i))/l(i)  ! Factor for direction to differentiate
    else
      y = y*k(x(i)/l(i))        ! Usual factor
    end if
  end do
end function


end module kernels
