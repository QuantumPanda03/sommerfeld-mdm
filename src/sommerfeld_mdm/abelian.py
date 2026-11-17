import numpy as np


def yukawa(alpha, mphi):

    V_inf = np.zeros((1, 1))

    def V_eff(r):
        return np.array([
            [-alpha * np.exp(-mphi*r) / r]
        ])

    return V_inf, V_eff
