import pytest
import numpy as np
from numpy.testing import assert_allclose

from sommerfeld_mdm.sommerfeld import sommerfeld


@pytest.mark.parametrize(
    "v_rel, expected_channels",
    [
        (0.001, [2]),        # Only neutral open
        (0.010, [1, 2]),     # Singly charged + neutral
        (0.030, [0, 1, 2])   # All channels open
    ]
)
def test_free_potential_open_channels(v_rel, expected_channels):

    M = 13700.0
    delta = 0.167

    V_inf = np.diag([8*delta, 2*delta, 0.0])

    def V_eff(r):
        return np.zeros((3, 3))

    T = sommerfeld(
        M,
        v_rel,
        V_inf,
        V_eff,
        x_inf=100.0
    )

    expected = np.zeros((3, 3), dtype=complex)

    expected[:, expected_channels] = (
        np.eye(3, dtype=complex)[:, expected_channels]
    )

    assert T.shape == (3, 3)

    assert_allclose(
        T,
        expected,
        rtol=1e-6,
        atol=1e-6
    )