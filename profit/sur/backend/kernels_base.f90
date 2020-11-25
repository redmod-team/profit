REAL*8 function kern_sqexp(r)
implicit none
REAL*8, intent(in) :: r
kern_sqexp = exp(-0.5d0*r**2)
end function
REAL*8 function dkern_sqexp(r)
implicit none
REAL*8, intent(in) :: r
dkern_sqexp = -r*exp(-0.5d0*r**2)
end function
REAL*8 function d2kern_sqexp(r)
implicit none
REAL*8, intent(in) :: r
d2kern_sqexp = (r**2 - 1)*exp(-0.5d0*r**2)
end function
REAL*8 function kern_matern32(r)
implicit none
REAL*8, intent(in) :: r
kern_matern32 = (sqrt(3.0d0)*r + 1)*exp(-1.7320508075688773d0*r)
end function
REAL*8 function dkern_matern32(r)
implicit none
REAL*8, intent(in) :: r
dkern_matern32 = -3*r*exp(-1.7320508075688773d0*r)
end function
REAL*8 function d2kern_matern32(r)
implicit none
REAL*8, intent(in) :: r
d2kern_matern32 = 3*(sqrt(3.0d0)*r - 1)*exp(-1.7320508075688773d0*r)
end function
REAL*8 function kern_matern52(r)
implicit none
REAL*8, intent(in) :: r
kern_matern52 = ((5.0d0/3.0d0)*r + sqrt(5.0d0)*r + 1)*exp( &
      -2.2360679774997897d0*r)
end function
REAL*8 function dkern_matern52(r)
implicit none
REAL*8, intent(in) :: r
dkern_matern52 = -1.0d0/3.0d0*(5*sqrt(5.0d0)*r + 15*r - 5)*exp( &
      -2.2360679774997897d0*r)
end function
REAL*8 function d2kern_matern52(r)
implicit none
REAL*8, intent(in) :: r
d2kern_matern52 = (5.0d0/3.0d0)*(5*r + 3*sqrt(5.0d0)*r - 2*sqrt(5.0d0) - &
      3)*exp(-2.2360679774997897d0*r)
end function
REAL*8 function kern_wend4(r)
implicit none
REAL*8, intent(in) :: r
kern_wend4 = (1 - r)**4*(4*r + 1)
end function
REAL*8 function dkern_wend4(r)
implicit none
REAL*8, intent(in) :: r
dkern_wend4 = 20*r*(r - 1)**3
end function
REAL*8 function d2kern_wend4(r)
implicit none
REAL*8, intent(in) :: r
d2kern_wend4 = (r - 1)**2*(80*r - 20)
end function
