import matplotlib.pyplot as plt

import sys
import numpy as np
import scipy.ndimage as ndi
from pathlib import Path


def com_label(ax):
    ax.set_ylabel(r"$y_\mathrm{com}$")


def phase_label(ax):
    ax.set_xlabel(r"$\phi$ / deg")


def smooth(phase, com, window):
    w = 10
    ycoms_smoothed = np.convolve(com, np.ones(w), "valid") / w
    phases_smoothed = np.convolve(phase, np.ones(w), "valid") / w

    return phases_smoothed, ycoms_smoothed


def mid_points(x):
    return x[:-1] + np.diff(x) / 2.0


def get_monotonic_intervals(phases, coms):
    # We want to find where the centres of mass are consistently rising and falling.
    deriv = np.diff(coms)

    rising_mask = deriv > 0
    falling_mask = deriv <= 0

    (indices,) = np.where(np.diff(rising_mask))
    indices += 1  # off by one otherwise.

    phase_turning_points = phases[indices]

    piecewise_monotonic_coms = np.split(coms, indices)
    piecewise_monotonic_com_phases = np.split(phases, indices)

    yield from zip(piecewise_monotonic_com_phases, piecewise_monotonic_coms)


def line(x, a0, a1):
    return a0 + a1 * x


def linear_fit(indep_var, dep_var, dep_var_err=None):
    popt, pcov = curve_fit(line, indep_var, dep_var)
    # popt, pcov = curve_fit(line, indep_var, dep_var, sigma=dep_var_err, absolute_sigma=True)
    perr = np.sqrt(np.diag(pcov))

    # Present as tuples
    a0 = popt[0], perr[0]  # y-intercept with error
    a1 = popt[1], perr[1]  # gradient with error

    return a0, a1


def plot_monotonic_sections(phase, com):
    fig, ax = plt.subplots()
    # ax.plot(phase, com, color="black", alpha=1)
    for phase, com in get_monotonic_intervals(phase, com):
        ax.plot(phase, com)


def get_longest_two_monotonic_intervals(phase, com):
    intervals = list(get_monotonic_intervals(phase, com))
    lengths = [len(x[0]) for x in intervals]

    *_, isecond, ifirst = np.argpartition(lengths, kth=len(lengths) - 1)

    # # Get first and second longest intervals
    # first, = np.where(np.argsort(lengths) == 0)
    # second, = np.where(np.argsort(lengths) == 1)

    return intervals[ifirst], intervals[isecond]


# def get_longest_falling_interval(phase, com):
#     from IPython import embed; embed()


def plot_longest_sections(phase, com, ax=None):
    # phasef, comf = get_longest_falling_interval(phase, com)
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)

    if ax is None:
        fig, ax = plt.subplots()
    ax.plot(longest[0], longest[1])
    ax.plot(second_longest[0], second_longest[1])


def get_truncated_longest_sections(phase, com, com_window_size):
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)
    phi1, com1 = longest
    phi2, com2 = second_longest

    com1_mid = com1.mean()
    com2_mid = com2.mean()

    com1_mid = 2330 / 2
    com2_mid = 2330 / 2

    mask1 = (com1 > (com1_mid - com_window_size)) & (com1 < (com1_mid + com_window_size))
    mask2 = (com2 > (com2_mid - com_window_size)) & (com2 < (com2_mid + com_window_size))

    return ((phi1[mask1], com1[mask1]), (phi2[mask2], com2[mask2]))


def plot_truncated_longest_sections(phase, com, ax=None, com_window_size=None):
    # phasef, comf = get_longest_falling_interval(phase, com)
    longest, second_longest = get_longest_two_monotonic_intervals(phase, com)

    if ax is None:
        fig, ax = plt.subplots()

    ((phi1, com1), (phi2, com2)) = get_truncated_longest_sections(phase, com, com_window_size=com_window_size)

    ax.plot(phi1, com1)
    ax.plot(phi2, com2)


def main(dname):
    npzdir = Path(dname)

    # from IPython import embed; embed()

    amplitude = float(dname.split("=")[1])

    phases = []
    ycoms = []
    images = []

    for f in npzdir.glob("*.npz"):
        ph = int(f.stem)
        if ph < -180:
            continue
        if ph >= 180:
            continue
        phases.append(ph)

    phases.sort()

    for phase in phases:
        fname = dname / Path(f"{phase}.npz")
        image = np.load(fname)["arr_0"]
        images.append(image)

    for image in images:
        ycoms.append(ndi.center_of_mass(image)[1])

    w = 5
    phases_smoothed, ycoms_smoothed = smooth(phases, ycoms, w)

    plot_monotonic_sections(phases_smoothed, ycoms_smoothed)

    fig, ax = plt.subplots()
    ax.plot(phases, ycoms, label="Raw Data")
    ax.plot(phases_smoothed, ycoms_smoothed, label="Smoothed")
    ax.set_ylabel(r"$y_\mathrm{com}$")
    ax.set_xlabel(r"$\phi$ / deg")
    ax.legend()

    fig, ax = plt.subplots()

    ax.plot(phases, ycoms, label="Raw Data", linestyle="--", alpha=0.75)
    ax.set_ylabel(r"$y_\mathrm{com}$")
    ax.set_xlabel(r"$\phi$ / deg")
    plot_longest_sections(phases_smoothed, ycoms_smoothed, ax=ax)
    ax.legend()

    fig, ax = plt.subplots()

    com_window_size = 20

    ax.plot(phases, ycoms, label="Raw Data", linestyle="--", alpha=0.2)
    ax.set_ylabel(r"$y_\mathrm{com}$")
    ax.set_xlabel(r"$\phi$ / deg")
    plot_truncated_longest_sections(phases_smoothed, ycoms_smoothed, ax=ax, com_window_size=com_window_size)
    ax.legend()

    frequency = 3e9

    # plt.show()

    # plot_lines_of_best_fit(phases_smoothed, ycoms_smoothed,
    #                        ax=ax, com_window_size=20)

    plt.show()


if __name__ == '__main__':
    main(sys.argv[1])
