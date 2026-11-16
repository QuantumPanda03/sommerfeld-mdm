import numpy as np
from numpy.testing import assert_allclose
from sommerfeld_mdm.sommerfeld import sommerfeld


def test_free_potential():

    M = 13700.0
    delta = 0.167
    v_rel = 0.001

    V_inf = np.diag([8*delta, 2*delta, 0.0])

    def V_eff(r):
        return np.zeros((3, 3))

    T = sommerfeld(M, v_rel, V_inf, V_eff, x_inf=0.5)

    assert_allclose(
        T,
        np.eye(3),
        rtol=1e-6,
        atol=1e-6
    )