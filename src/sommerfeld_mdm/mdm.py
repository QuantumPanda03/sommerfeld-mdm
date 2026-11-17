import numpy as np


def mdm_quintupletQ0S0(M, delta=0.167):
    """
    Fermionic SU(2) quintuplet, Y=0.
    Q=0, S=0, L=0 sector.

    Basis:
        0: chi++ chi--
        1: chi+  chi-
        2: chi0  chi0

    LO potential and annihilation matrix:
    Cirelli, Strumia & Tamburini (2007), eq. (16).

    All masses are in GeV.
    """

    # Electroweak parameters
    alpha_em = 1 / 127.95
    sin2_thetaW = 0.23122
    cos2_thetaW = 1 - sin2_thetaW

    alpha2 = alpha_em / sin2_thetaW

    mW = 80.377
    mZ = 91.1876

    # Asymptotic mass splittings
    V_inf = np.diag([
        8 * delta,
        2 * delta,
        0.0
    ])

    # Interaction potential
    def V_eff(r):

        A = (
            alpha_em / r
            + alpha2 * cos2_thetaW * np.exp(-mZ * r) / r
        )

        B = alpha2 * np.exp(-mW * r) / r

        # Potential matrix: Cirelli et al., eq. (16)
        return np.array([
            [-4*A,        -2*B,           0.0],
            [-2*B,        -A,    -3*np.sqrt(2)*B],
            [0.0, -3*np.sqrt(2)*B,           0.0]
        ])

    # Annihilation matrix: Cirelli et al., eq. (16)
    Gamma = (
        3 * np.pi * alpha2**2 / (25 * M**2)
        * np.array([
            [12,              6,     2*np.sqrt(2)],
            [6,               9,     5*np.sqrt(2)],
            [2*np.sqrt(2), 5*np.sqrt(2),     6]
        ])
    )

    return V_inf, V_eff, Gamma