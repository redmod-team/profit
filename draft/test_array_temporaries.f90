module modu
    implicit none

    type :: t
        real(8) :: x(100, 100), y(100)
    end type
end module modu


program test
    use modu
    implicit none
    integer, parameter :: n = 100
    !real(8) :: x(n), y(n)
    real(8), allocatable :: x(:,:), y(:)
    type(t) :: ti

    ti%x = cos(ti%x) + sin(cos(ti%x))

    allocate(x(n,n), y(n))
    x = 1.0d0


    ti%x = 1.0d0
    !call evil(x, y, func)
    print *, ti%y(1)

    ! SIMD
    ! real(8) :: temp(n)
    ! do k = 1,n
    !   temp(k) = sin(x(k))
    ! end do
    ! do k = 1,n
    !   y(k) = cos(temp(k))
    ! end do
    !
    ! SIMD without TEMP
    ! do k = 1,n
    !   y(k) = sin(x(k))
    ! end do
    ! do k = 1,n
    !   y(k) = cos(y(k))
    ! end do
    !
    ! NON-SIMD
    ! do k = 1,n
    !   y(k) = cos(sin(x(k)))
    ! end do
    !
    ! y(:) = x(:)*f(z(:))
    ! y(:) = t%x(:)*f(t2%z(:))
    !
    ! -heap-arrays
    !
    deallocate(x, y)

    contains


    ! subroutine func(a, b)
    !     real(8) :: a(:), b(:)
    !     b = cos(a) + sin(cos(a))
    ! end subroutine func

    ! subroutine evil(a, b, func2)
    !     real(8) :: a(:, :), b(:)
    !     external func2
    !     call func2(a(10, :), b)
    ! end subroutine evil

end program test
