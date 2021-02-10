"""Tests for Fortran UQ code from Roland Preuss
"""
import os
import pytest
import numpy as np
from time import time
from profit import uq


@pytest.fixture(scope='module')
def uqp():
    from fffi import FortranModule

    cwd = os.path.dirname(__file__)
    os.chdir(cwd)

    fort_mod = FortranModule('uqp', 'mod_unqu')

    fort_mod.fdef("""
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

    fort_mod.compile(verbose=1)
    fort_mod.load()
    return fort_mod


@pytest.mark.skip(reason="Fortran code for UQ currently unavailable")
def test_hermite_points(uqp):
    """
    Test Hermite quadrature points in 2D
    """

    print('=== Hermite 2D quadrature points ===')

    mean1 = 5.0
    std1 = 0.1
    mean2 = 2.0
    std2 = 0.2

    t = time()  # UQP timing
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
    axi = np.zeros((uqp.nall, uqp.npar), order='F')
    uqp.pre_uq(axi)
    print('UQP: {:4.2f} ms'.format(1000*(time() - t)))

    t = time()  # ChaosPy timing
    uqu = uq.UQ()
    uqu.backend = uq.backend.ChaosPy(order=3, sparse=False)
    uqu.params['u'] = uqu.backend.Normal(mean1, std1)
    uqu.params['v'] = uqu.backend.Normal(mean2, std2)
    axi2 = uqu.get_eval_points().T
    print('ChaosPy: {:4.2f} ms'.format(1000*(time() - t)))

    axis = axi[np.lexsort((axi[:, 0], axi[:, 1]))]
    axi2s = axi2[np.lexsort((axi2[:, 0], axi2[:, 1]))]

    assert np.allclose(axis, axi2s)
    print('Success: UQP and ChaosPy results match')
