from typing import Any

import numpy as np
from scipy.optimize import curve_fit


ValueWithErrorT = tuple[float, float]

def line(x, a0, a1) -> Any:
    return a0 + a1 * x


def linear_fit(indep_var, dep_var, dep_var_err) -> tuple[ValueWithErrorT, ValueWithErrorT]:
    popt, pcov = curve_fit(line, indep_var, dep_var, sigma=dep_var_err, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    # Present as tuples
    a0 = popt[0], perr[0] # y-intercept with error
    a1 = popt[1], perr[1] # gradient with error

    return a0, a1

