REAL*8 function kern_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
kern_num = sin(sqrt((x_a - x_b)**2/lx**2))
end function
REAL*8 function dkdx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx_num = sqrt((x_a - x_b)**2/lx**2)*cos(sqrt((x_a - x_b)**2/lx**2))/( &
      x_a - x_b)
end function
INTEGER*4 function dkdy_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdy_num = 0
end function
REAL*8 function dkdx0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx0_num = -sqrt((x_a - x_b)**2/lx**2)*cos(sqrt((x_a - x_b)**2/lx**2))/ &
      (x_a - x_b)
end function
INTEGER*4 function dkdy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdy0_num = 0
end function
REAL*8 function d2kdxdx0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdxdx0_num = sin(sqrt((x_a - x_b)**2/lx**2))/lx**2
end function
INTEGER*4 function d2kdydy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdydy0_num = 0
end function
INTEGER*4 function d2kdxdy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdxdy0_num = 0
end function
