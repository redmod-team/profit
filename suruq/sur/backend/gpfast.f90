module gpfast
  implicit none
contains

  function kern(x0, x1, a)
    real(8) :: x0(:), x1(:), a, kern
    kern = exp(-sqrt(sum((x1-x0)**2))/(2*a**2))
  end function kern

  subroutine gp_matrix(x0, x1, a, K)
    real(8) :: x0(:,:), x1(:,:), a, K(:,:)
    integer :: k0, k1, n0, n1
    n0 = size(x0)
    n1 = size(x1)
    do k1 = 1, n1
      do k0 = 1, n0
        K(k0, k1) = kern(x0(:,k0), x1(:,k1), a)
      end do
    end do
  end subroutine gp_matrix

end module gpfast

