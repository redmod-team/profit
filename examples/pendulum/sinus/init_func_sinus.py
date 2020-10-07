from sympy import *
from sympy.utilities.autowrap import autowrap, ufuncify
from sympy.utilities.lambdify import lambdastr, lambdify, implemented_function
import shutil

tmp = '.'
#try:
#    shutil.rmtree(tmp)
#except:
#    pass

#%% # Kernel functions (kernel, 1st and second derivatives)
x, y, xa, ya, xb, yb, l, lx, ly = symbols(r'x y x_a y_a x_b, y_b, l, lx, ly')


# kern_sqexp = exp(-x**2/(2*l**2))
# kern_sqexp_sin = exp(-sin(x/2)**2/(2*l**2))

# kern_y = kern_sqexp.subs(x, ya-yb).subs(l, ly)
# kern_x = kern_sqexp.subs(x, xa-xb).subs(l, lx)

# kern_sqexp = exp(-x**2/(2*l**2))
# kern_sqexp_sin = exp(-sin(x/2)**2/(2*l**2))

# kern_y = kern_sqexp.subs(x, ya-yb).subs(l, ly)
# kern_x = kern_sqexp_sin.subs(x, xa-xb).subs(l, lx)

# kern_sin1 = sin(x)
# kern_sin2 = sin(x)

# kern_y = kern_sin1.subs(x, ya-yb).subs(l, ly)
# kern_x = kern_sin2.subs(x, xa-xb).subs(l, lx)

# kern = (kern_x).simplify()#*kern_y).simplify()

kern_sin = sin(x)
# kern_x = kern_cos.subs(x, sqrt((xa-xb)**2/lx**2))
kern_x = kern_sin.subs(x, sqrt((xa-xb)**2/lx**2))#+(ya-yb)**2/ly**2))
# kern_y = kern_cos.subs(x, ya-yb).subs(l, ly)
kern = (kern_x).simplify()
# print(kern)

dkdxa = diff(kern, xa).simplify()
# print(dkdxa,'\n')
dkdxb = diff(kern, xb).simplify()
dkdya = diff(kern, ya).simplify()
dkdyb = diff(kern, yb).simplify()
dkdxadxb = diff(kern, xa, xb).simplify()
dkdyadyb = diff(kern, ya, yb).simplify()
dkdxadyb = diff(kern, xa, yb).simplify()
print(dkdxadxb)

# d3kdxdx0dy0 = diff(kern, xa, xb, yb).simplify()
# d3kdydy0dy0 = diff(kern, ya, yb, yb).simplify()
# d3kdxdy0dy0 = diff(kern, xa, yb, yb).simplify()

# dkdlx = diff(kern, lx).simplify()
# dkdly = diff(kern, ly).simplify()

# d3kdxdx0dlx = diff(kern, xa, xb, lx).simplify()
# d3kdydy0dlx = diff(kern, ya, yb, lx).simplify()
# d3kdxdy0dlx = diff(kern, xa, yb, lx).simplify()

# d3kdxdx0dly = diff(kern, xa, xb, ly).simplify()
# d3kdydy0dly = diff(kern, ya, yb, ly).simplify()
# d3kdxdy0dly = diff(kern, xa, yb, ly).simplify()

#%%
from sympy.utilities.codegen import codegen
seq = (xa, ya, xb, yb, lx, ly)
[(name, code), (h_name, header)] = \
codegen([('kern_num', kern),
          ('dkdx_num', dkdxa),
          ('dkdy_num', dkdya),
          ('dkdx0_num', dkdxb),
          ('dkdy0_num', dkdyb),
          ('d2kdxdx0_num', dkdxadxb),
          ('d2kdydy0_num', dkdyadyb),
          ('d2kdxdy0_num', dkdxadyb),
          # ('d3kdxdx0dy0_num', d3kdxdx0dy0),
          # ('d3kdydy0dy0_num', d3kdydy0dy0),
          # ('d3kdxdy0dy0_num', d3kdxdy0dy0),
          # ('dkdlx_num', dkdlx),
          # ('dkdly_num', dkdly),
          # ('d3kdxdx0dlx_num', d3kdxdx0dlx),
          # ('d3kdydy0dlx_num', d3kdydy0dlx),
          # ('d3kdxdy0dlx_num', d3kdxdy0dlx),
          # ('d3kdxdx0dly_num', d3kdxdx0dly),
          # ('d3kdydy0dly_num', d3kdydy0dly),
          # ('d3kdxdy0dly_num', d3kdxdy0dly),
          ], "F95", "kernels",
        argument_sequence=seq,
        header=False, empty=False)
with open(name, 'w') as f:
    f.write(code)
