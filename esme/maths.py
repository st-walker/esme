from typing import Any

import numpy as np
from scipy.optimize import curve_fit

ValueWithErrorT = tuple[float, float]


def line(x, a0, a1) -> Any:
    return a0 + a1 * x


def linear_fit(
    indep_var, dep_var, dep_var_err
) -> tuple[ValueWithErrorT, ValueWithErrorT]:
    absolute_sigma = True
    if dep_var_err is None:
        absolute_sigma = False
    try:
        popt, pcov = curve_fit(
            line, indep_var, dep_var, sigma=dep_var_err, absolute_sigma=absolute_sigma
        )
    except:
        import ipdb; ipdb.set_trace()
    perr = np.sqrt(np.diag(pcov))

    # Present as tuples
    a0 = popt[0], perr[0]  # y-intercept with error
    a1 = popt[1], perr[1]  # gradient with error

    return a0, a1


def gauss(x, a, mu, sigma) -> Any:
    return a * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


# def parabola(x, a, b, c):
#     return a*x**2 + b*x + c

def parabola(x, a, b, c):
    return a*x**2 + b*x + c

def get_gaussian_fit(x, y) -> tuple[tuple, tuple]:
    """popt/perr order: a, mu, sigma"""
    mu0 = y.argmax()
    a0 = y.max()
    sigma0 = 1

    # Bounds argument of curve_fit slows the fitting procedure down too much
    # (>2x worse), so avoid using it here.
    popt, pcov = curve_fit(
        gauss,
        x,
        y,
        p0=[a0, mu0, sigma0],
    )
    variances = np.diag(pcov)
    if (variances < 0).any():
        raise RuntimeError(f"Negative variance detected: {variances}")
    perr = np.sqrt(variances)
    return popt, perr
