"""
Created: 2019-04-01
@author: Christopher Albert <albert@alumni.tugraz.at>

Compiles CFF for test_arrays
"""

import os
import numpy as np
from unittest import TestCase
from fffi import fortran_module
from suruq import uq


class TestBackendUQP(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.origcwd = os.getcwd()
        cls.cwd = os.path.dirname(__file__)
        os.chdir(cls.cwd)

        try:
            cls.uqp = fortran_module('uqp', 'mod_unqu', path=cls.cwd)

            cls.uqp.fdef("""
              integer :: np, nall, npar, nt, iflag_run, iflag_mod, iflag_pol  
            
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
            
            cls.uqp.compile(verbose=1)
            cls.uqp.load()
        
        except BaseException:
            os.chdir(cls.origcwd)
            raise        
        

    @classmethod
    def tearDownClass(cls):
        os.chdir(cls.origcwd)
        

    def test_hermite_points(self):
        """
        Test Hermite quadrature points in 2D
        """
        mean1 = 5.0
        std1 = 0.1
        mean2 = 2.0
        std2 = 0.2

        uqp = self.uqp
        
        uqp.npar = 2
        uqp.allocate_params()
        
        uqp.np = 3
        uqp.nt = 1
        uqp.iflag_run = 1
        uqp.iflag_mod = 2
        uqp.iflag_pol = 2  # 1: Legende, 2: Hermite, 3: Laguerre
        uqp.set_hermite_mean_std(1, mean1, std1)
        uqp.set_hermite_mean_std(2, mean2, std2)
        uqp.init_uq()
        
        axi = np.zeros((uqp.nall,uqp.npar), order='F')
        uqp.pre_uq(axi)
        
        uq.backend = uq.ChaosPy(order = 3, sparse = False)
        
        uq.params['u'] = uq.Normal(mean1, std1)
        uq.params['v'] = uq.Normal(mean2, std2)
        
        axis = axi[np.lexsort((axi[:,0], axi[:,1]))]
        
        axi2 = uq.get_eval_points().T
        axi2s = axi2[np.lexsort((axi2[:,0], axi2[:,1]))]
        
        np.testing.assert_almost_equal(axis, axi2s)

if __name__ == "__main__":
    test = TestBackendUQP()
    test.setUpClass()
    test.test_hermite_points()
    test.tearDownClass()
