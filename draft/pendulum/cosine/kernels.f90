REAL*8 function kern_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
kern_num = cos(sqrt((y_a - y_b)**2/ly**2 + (x_a - x_b)**2/lx**2))
end function
REAL*8 function dkdx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx_num = -(x_a - x_b)*sin(sqrt((lx**2*(y_a - y_b)**2 + ly**2*(x_a - &
      x_b)**2)/(lx**2*ly**2)))/(lx**2*sqrt((lx**2*(y_a - y_b)**2 + ly** &
      2*(x_a - x_b)**2)/(lx**2*ly**2)))
end function
REAL*8 function dkdy_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdy_num = -(y_a - y_b)*sin(sqrt((lx**2*(y_a - y_b)**2 + ly**2*(x_a - &
      x_b)**2)/(lx**2*ly**2)))/(ly**2*sqrt((lx**2*(y_a - y_b)**2 + ly** &
      2*(x_a - x_b)**2)/(lx**2*ly**2)))
end function
REAL*8 function dkdx0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx0_num = (x_a - x_b)*sin(sqrt((lx**2*(y_a - y_b)**2 + ly**2*(x_a - &
      x_b)**2)/(lx**2*ly**2)))/(lx**2*sqrt((lx**2*(y_a - y_b)**2 + ly** &
      2*(x_a - x_b)**2)/(lx**2*ly**2)))
end function
REAL*8 function dkdy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdy0_num = (y_a - y_b)*sin(sqrt((lx**2*(y_a - y_b)**2 + ly**2*(x_a - &
      x_b)**2)/(lx**2*ly**2)))/(ly**2*sqrt((lx**2*(y_a - y_b)**2 + ly** &
      2*(x_a - x_b)**2)/(lx**2*ly**2)))
end function
REAL*8 function d2kdxdx0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdxdx0_num = ly**2*(lx**2*ly**2*sqrt((lx**2*(y_a - y_b)**2 + ly**2*( &
      x_a - x_b)**2)/(lx**2*ly**2))*(-(x_a - x_b)**2 + (lx**2*(y_a - &
      y_b)**2 + ly**2*(x_a - x_b)**2)/ly**2)*sin(sqrt((lx**2*(y_a - y_b &
      )**2 + ly**2*(x_a - x_b)**2)/(lx**2*ly**2))) + (x_a - x_b)**2*(lx &
      **2*(y_a - y_b)**2 + ly**2*(x_a - x_b)**2)*cos(sqrt((lx**2*(y_a - &
      y_b)**2 + ly**2*(x_a - x_b)**2)/(lx**2*ly**2))))/(lx**2*(lx**2*( &
      y_a - y_b)**2 + ly**2*(x_a - x_b)**2)**2)
end function
REAL*8 function d2kdydy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdydy0_num = lx**2*(lx**2*ly**2*sqrt((lx**2*(y_a - y_b)**2 + ly**2*( &
      x_a - x_b)**2)/(lx**2*ly**2))*(-(y_a - y_b)**2 + (lx**2*(y_a - &
      y_b)**2 + ly**2*(x_a - x_b)**2)/lx**2)*sin(sqrt((lx**2*(y_a - y_b &
      )**2 + ly**2*(x_a - x_b)**2)/(lx**2*ly**2))) + (y_a - y_b)**2*(lx &
      **2*(y_a - y_b)**2 + ly**2*(x_a - x_b)**2)*cos(sqrt((lx**2*(y_a - &
      y_b)**2 + ly**2*(x_a - x_b)**2)/(lx**2*ly**2))))/(ly**2*(lx**2*( &
      y_a - y_b)**2 + ly**2*(x_a - x_b)**2)**2)
end function
REAL*8 function d2kdxdy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdxdy0_num = (x_a - x_b)*(y_a - y_b)*(cos(sqrt((y_a - y_b)**2/ly**2 + &
      (x_a - x_b)**2/lx**2))/((y_a - y_b)**2/ly**2 + (x_a - x_b)**2/lx &
      **2) - sin(sqrt((y_a - y_b)**2/ly**2 + (x_a - x_b)**2/lx**2))/(( &
      y_a - y_b)**2/ly**2 + (x_a - x_b)**2/lx**2)**(3.0d0/2.0d0))/(lx** &
      2*ly**2)
end function
