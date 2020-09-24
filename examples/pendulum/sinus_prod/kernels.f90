REAL*8 function kern_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
kern_num = sin(x_a - x_b)
end function
REAL*8 function dkdx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdx_num = cos(x_a - x_b)
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
dkdx0_num = -cos(x_a - x_b)
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
d2kdxdx0_num = sin(x_a - x_b)
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
INTEGER*4 function d3kdxdx0dy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdx0dy0_num = 0
end function
INTEGER*4 function d3kdydy0dy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdydy0dy0_num = 0
end function
INTEGER*4 function d3kdxdy0dy0_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdy0dy0_num = 0
end function
INTEGER*4 function dkdlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdlx_num = 0
end function
INTEGER*4 function dkdly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
dkdly_num = 0
end function
INTEGER*4 function d3kdxdx0dlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdx0dlx_num = 0
end function
INTEGER*4 function d3kdydy0dlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdydy0dlx_num = 0
end function
INTEGER*4 function d3kdxdy0dlx_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdy0dlx_num = 0
end function
INTEGER*4 function d3kdxdx0dly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdx0dly_num = 0
end function
INTEGER*4 function d3kdydy0dly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdydy0dly_num = 0
end function
INTEGER*4 function d3kdxdy0dly_num(x_a, y_a, x_b, y_b, lx, ly)
implicit none
REAL*8, intent(in) :: x_a
REAL*8, intent(in) :: y_a
REAL*8, intent(in) :: x_b
REAL*8, intent(in) :: y_b
REAL*8, intent(in) :: lx
REAL*8, intent(in) :: ly
d3kdxdy0dly_num = 0
end function
