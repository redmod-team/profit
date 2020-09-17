REAL*8 function kern_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
kern_num = exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin(x_a - x_b)**2/lx &
      **2)
end function
REAL*8 function dkdx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx_num = -exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin(x_a - x_b)**2/lx &
      **2)*sin(x_a - x_b)*cos(x_a - x_b)/lx**2
end function
REAL*8 function dkdy_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdy_num = (-y_a + y_b)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(x_a &
      - x_b)**2)/(lx**2*ly**2))/ly**2
end function
REAL*8 function dkdx0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx0_num = exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin(x_a - x_b)**2/lx &
      **2)*sin(x_a - x_b)*cos(x_a - x_b)/lx**2
end function
REAL*8 function dkdy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdy0_num = (y_a - y_b)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(x_a &
      - x_b)**2)/(lx**2*ly**2))/ly**2
end function
REAL*8 function d2kdxdx0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdxdx0_num = (lx**2*cos(2.0d0*x_a - 2.0d0*x_b) - sin(x_a - x_b)**2*cos &
      (x_a - x_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(x_a &
      - x_b)**2)/(lx**2*ly**2))/lx**4
end function
REAL*8 function d2kdydy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdydy0_num = (ly**2 - (y_a - y_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 &
      + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))/ly**4
end function
REAL*8 function d2kdxdy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d2kdxdy0_num = -(y_a - y_b)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin &
      (x_a - x_b)**2)/(lx**2*ly**2))*sin(x_a - x_b)*cos(x_a - x_b)/(lx &
      **2*ly**2)
end function
REAL*8 function d3kdxdx0dy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdx0dy0_num = (y_a - y_b)*(lx**2*cos(2.0d0*x_a - 2.0d0*x_b) - sin( &
      x_a - x_b)**2*cos(x_a - x_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 &
      + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))/(lx**4*ly**2)
end function
REAL*8 function d3kdydy0dy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdydy0dy0_num = (3*ly**2 - (y_a - y_b)**2)*(y_a - y_b)*exp(-0.5d0*(lx &
      **2*(y_a - y_b)**2 + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))/ly** &
      6
end function
REAL*8 function d3kdxdy0dy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdy0dy0_num = (ly**2 - (y_a - y_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b) &
      **2 + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))*sin(x_a - x_b)*cos( &
      x_a - x_b)/(lx**2*ly**4)
end function
REAL*8 function dkdlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdlx_num = exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin(x_a - x_b)**2/lx &
      **2)*sin(x_a - x_b)**2/lx**3
end function
REAL*8 function dkdly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdly_num = (y_a - y_b)**2*exp(-0.5d0*(y_a - y_b)**2/ly**2 - 0.5d0*sin( &
      x_a - x_b)**2/lx**2)/ly**3
end function
REAL*8 function d3kdxdx0dlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdx0dlx_num = (-2*lx**4*cos(2.0d0*x_a - 2.0d0*x_b) + lx**2*(3*cos( &
      2.0d0*x_a - 2.0d0*x_b) + 2)*sin(x_a - x_b)**2 - sin(x_a - x_b)**4 &
      *cos(x_a - x_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin( &
      x_a - x_b)**2)/(lx**2*ly**2))/lx**7
end function
REAL*8 function d3kdydy0dlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdydy0dlx_num = (ly**2 - (y_a - y_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b) &
      **2 + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))*sin(x_a - x_b)**2/( &
      lx**3*ly**4)
end function
REAL*8 function d3kdxdy0dlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdy0dlx_num = (2*lx**2 - sin(x_a - x_b)**2)*(y_a - y_b)*exp(-0.5d0*( &
      lx**2*(y_a - y_b)**2 + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))* &
      sin(x_a - x_b)*cos(x_a - x_b)/(lx**5*ly**2)
end function
REAL*8 function d3kdxdx0dly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdx0dly_num = (y_a - y_b)**2*(lx**2*cos(2.0d0*x_a - 2.0d0*x_b) - sin &
      (x_a - x_b)**2*cos(x_a - x_b)**2)*exp(-0.5d0*(lx**2*(y_a - y_b)** &
      2 + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))/(lx**4*ly**3)
end function
REAL*8 function d3kdydy0dly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdydy0dly_num = (-2*ly**4 + 5*ly**2*(y_a - y_b)**2 - (y_a - y_b)**4)* &
      exp(-0.5d0*(lx**2*(y_a - y_b)**2 + ly**2*sin(x_a - x_b)**2)/(lx** &
      2*ly**2))/ly**7
end function
REAL*8 function d3kdxdy0dly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdy0dly_num = (2*ly**2 - (y_a - y_b)**2)*(y_a - y_b)*exp(-0.5d0*(lx &
      **2*(y_a - y_b)**2 + ly**2*sin(x_a - x_b)**2)/(lx**2*ly**2))*sin( &
      x_a - x_b)*cos(x_a - x_b)/(lx**2*ly**5)
end function
