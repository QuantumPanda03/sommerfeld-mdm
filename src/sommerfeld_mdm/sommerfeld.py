import numpy as np
from scipy.integrate import solve_ivp

def sommerfeld(M, v_rel, V_inf, V_eff, x_inf=100.0):

    V_inf = np.asarray(V_inf)
    n = V_inf.shape[0]

    I = np.eye(n, dtype=complex)

    if V_inf.shape != (n, n):
        raise ValueError("V_inf must be a square matrix.")

    if not np.allclose(V_inf, np.diag(np.diag(V_inf))):
        raise ValueError("V_inf must be diagonal.")

    if M <= 0:
        raise ValueError("M must be positive.")

    if not 0 < v_rel < 1:
        raise ValueError("v_rel must be in (0, 1).")

    # Beneke conventions
    v = v_rel / 2.0
    E = M * v**2

    # Dimensionless wave numbers
    k = np.sqrt(1.0 - np.diag(V_inf)/E + 0j)
    K = np.diag(k)

    # Initial conditions
    x0 = 1e-9

    N0 = x0 * I
    P0 = I.copy()

    y0 = np.concatenate([N0.ravel(), P0.ravel()])

    def rhs(x, y):

        N = y[:n*n].reshape(n, n)
        P = y[n*n:].reshape(n, n)

        r = x / (M*v)
        W = np.asarray(V_eff(r), dtype=complex) / E

        # Eq. (110)
        dN = I + 1j*(K @ N + N @ K) - N @ W @ N

        # Eq. (111)
        Z = -1j*K + W @ N

        # Eq. (116)
        dP = -P @ Z

        return np.concatenate([dN.ravel(), dP.ravel()])

    sol = solve_ivp(
        rhs,
        (x0, x_inf),
        y0,
        method="BDF",
        rtol=1e-10,
        atol=1e-12
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # Recover P at x_inf
    # Extract the solution at the final integration point
    y_final = sol.y[:, -1]

    # Extract the matrix P
    P_flat = y_final[n*n:]

    # Reshape P into an n x n matrix
    P = P_flat.reshape(n, n)

    # Eq. (117)
    thresholds = np.diag(V_inf)
    open_channels = E > np.diag(V_inf)

    T = np.zeros((n, n), dtype=complex)

    T[:, open_channels] = (
        P[:, open_channels]
        * np.exp(-1j * k[open_channels] * x_inf)[None, :]
    )

    return T