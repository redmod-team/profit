"""
Created: Tue Mar 26 09:45:46 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""
from profit import uq
import numpy as np
from fffi import FortranModule

from time import time

mean1 = 5.0
std1 = 0.1
mean2 = 2.0
std2 = 0.2

uqp = FortranModule('uqp', 'mod_unqu')

uqp.fdef("""
  integer :: nporder, nall, npar, nt, iflag_run, iflag_mod, iflag_pol

  subroutine allocate_params end
  subroutine init_uq end
  
  subroutine set_legendre_borders(kpar, lower, upper)
    integer, intent(in) :: kpar
    double precision, intent(in) :: lower, upper
  end

  subroutine set_hermite_mean_std(kpar, mean0, std)
    integer, intent(in) :: kpar
    double precision, intent(in) :: mean0, std
  end
  
  subroutine pre_uq(axi)
    double precision, intent(inout) :: axi(:,:)
  end
  
  subroutine run_uq end
  """)

uqp.compile(verbose=1)
uqp.load()
#%%
print('=== Hermite 2D quadrature points ===')

t = time()  # UQP timing
uqp.npar = 2
uqp.allocate_params()

uqp.nporder = 30
uqp.nt = 1
uqp.iflag_run = 1
uqp.iflag_mod = 2
uqp.iflag_pol = 2  # 1: Legende, 2: Hermite, 3: Laguerre
uqp.set_hermite_mean_std(1, mean1, std1)
uqp.set_hermite_mean_std(2, mean2, std2)
uqp.init_uq()

axi = np.zeros((uqp.nall,uqp.npar), order='F')
uqp.pre_uq(axi)

print('UQP: {:4.2f} ms'.format(1000*(time() - t)))

t = time()  # ChaosPy timing
uq.backend = uq.ChaosPy(order = uqp.nporder, sparse = False)
uq.params['u'] = uq.Normal(mean1, std1)
uq.params['v'] = uq.Normal(mean2, std2)
axi2 = uq.get_eval_points().T

print('ChaosPy: {:4.2f} ms'.format(1000*(time() - t)))

axis = axi[np.lexsort((axi[:,0], axi[:,1]))]
axi2s = axi2[np.lexsort((axi2[:,0], axi2[:,1]))]

np.testing.assert_almost_equal(axis, axi2s)
print('Success: UQP and ChaosPy results match')

# TODO: move this into unit test
