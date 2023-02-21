import logging

import matplotlib.pyplot as plt
import latdraw
import pandas as pd
from ocelot.cpbd.beam import Twiss

from . import sim
from .sections import sections
from . import lattice


LOG = logging.getLogger(__name__)

MAD8 = "/Users/stuartwalker/physics/gpt-csr/mad8DesyPublic/TWISS_I1D"


Z_LABEL_STRING = r"$z$ / m"
E_LABEL_STRING = r"$E$ / MeV"
BETA_LABEL_STRING = r"$\beta$ / m"
BETX_LABEL_STRING = r"$\beta_x$ / m"
BETY_LABEL_STRING = r"$\beta_y$ / m"
ALFX_LABEL_STRING = r"$\alpha_x$"
ALFY_LABEL_STRING = r"$\alpha_y$"
DX_LABEL_STRING = "$D_x$ / m"
D_LABEL_STRING = "$D$ / m"


# Do I need to also simulate TDS Scan?


from ocelot.cpbd.optics import twiss as twissc, Twiss

# So we treat SectionedFEL as as "navigator" factory I guess?
# Track to this point.  Then track to that one using this previous parray.  And so on.

def _save_fig_or_show(fig, outdir, name):
    if outdir is None:
        plt.show()
    else:
        fname = outdir / name
        LOG.info(f"wrote {fname}")
        fig.savefig(fname)

def cathode_to_first_a1_cavity(outdir=None):
    """AKA cathode to the where ASTRA tracking ends and OCELOT begins"""
    all_twiss, mlat = sim.cathode_to_first_a1_cavity_optics()

    s_offset = all_twiss.iloc[0].s

    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axt, axe) = latdraw.subplots_with_lattice(bl, nrows=2)

    axt.plot(all_twiss.s, all_twiss.beta_x, label="$x$")
    axt.plot(all_twiss.s, all_twiss.beta_y, label="$y$")

    axt.set_ylabel(BETA_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)

    axt.legend()

    axe.plot(all_twiss.s, all_twiss.E * 1e3)
    axe.set_xlabel(Z_LABEL_STRING)

    mx.set_title("Cathode to first A1 Cavity")

    _save_fig_or_show(fig, outdir, "cathode-to-first-a1-cavity-design-optics.pdf")



def a1_to_i1d_design_optics(outdir=None):
    all_twiss, mlat = sim.a1_to_i1d_design_optics()

    s_offset = all_twiss.iloc[0].s

    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axt, axd, axe) = latdraw.subplots_with_lattice(bl, nrows=3)

    axt.plot(all_twiss.s, all_twiss.beta_x, label="$x$")
    axt.plot(all_twiss.s, all_twiss.beta_y, label="$y$")
    axd.plot(all_twiss.s, all_twiss.Dx, label="x")
    axd.plot(all_twiss.s, all_twiss.Dy, label="y")

    axd.set_ylabel(D_LABEL_STRING)
    # axd.legend()

    axt.set_ylabel(BETA_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)

    axt.legend()

    axe.plot(all_twiss.s, all_twiss.E * 1e3)
    axe.set_xlabel(Z_LABEL_STRING)

    mx.set_title("First A1 Cavity to I1D Design Optics")

    _save_fig_or_show(fig, outdir, "a1-to-i1d-design-optics.pdf")


def a1_to_q52_matching_point_measurement_optics(outdir=None):
    all_twiss, mlat = sim.a1_to_q52_matching_point_measurement_optics()

    s_offset = all_twiss.iloc[0].s

    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axt, axd, axe) = latdraw.subplots_with_lattice(bl, nrows=3)

    axt.plot(all_twiss.s, all_twiss.beta_x, label="$x$")
    axt.plot(all_twiss.s, all_twiss.beta_y, label="$y$")
    axd.plot(all_twiss.s, all_twiss.Dx, label="x")
    axd.plot(all_twiss.s, all_twiss.Dy, label="y")

    axd.set_ylabel(D_LABEL_STRING)
    # axd.legend()

    axt.set_ylabel(BETA_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)

    axt.legend()

    axe.plot(all_twiss.s, all_twiss.E * 1e3)
    axe.set_xlabel(Z_LABEL_STRING)

    mx.set_title("First A1 Cavity to QI.52.I1 Measurement Optics")

    _save_fig_or_show(fig, outdir, "a1-to-i1d-design-optics.pdf")



def qi52_to_i1d_dscan_optics(dscan_conf, outdir=None):
    # This bit is just for getting the design machine instance for the
    # plot, we don't use the design optics otherwise.
    design_twiss, mlat = sim.a1_to_i1d_design_optics()
    s_offset = design_twiss.iloc[0].s
    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])

    fig, (mx, axt, axd) = latdraw.subplots_with_lattice(bl, nrows=2)

    gen = sim.qi52_matching_point_to_i1d_measurement_optics(dscan_conf)
    for i, (dispersion, twiss, _) in enumerate(gen):
        axd.plot(twiss.s, twiss.Dx, label=fr"$D_x\,=\,{dispersion}\,\mathrm{{m}}$")

        (line,) = axt.plot(twiss.s, twiss.beta_x)

    axd.set_xlabel(Z_LABEL_STRING)
    axd.set_ylabel(DX_LABEL_STRING)
    axt.set_ylabel(BETX_LABEL_STRING)


    screen_s = twiss.s[twiss.id == "OTRC.64.I1D"].item()
    vlargs = {"x": screen_s, "label": "OTRC.64.I1D", "linestyle": ":", "color": "black"}
    axt.axvline(**vlargs)
    del vlargs["label"]
    axd.axvline(**vlargs)

    max_beta_x_up_to_screen = max(twiss.set_index(keys="id").loc[:"OTRC.64.I1D"].beta_x)
    mx.set_xlim(twiss.iloc[0].s, twiss.iloc[-1].s)
    axt.set_ylim(axt.get_ylim()[0], max_beta_x_up_to_screen * 1.1)


    mx.set_title("QI.52.I1 to I1D Dispersion Scan Optics")
    axt.legend()
    axd.legend()

    _save_fig_or_show(fig, outdir, "i1d-dscan-optics.pdf")


def a1_to_i1d_piecewise_measurement_optics(dscan_conf, outdir=None):
    full_sequence = lattice.make_to_i1d_lattice().get_sequence()

    all_twiss, mlat = sim.a1_to_q52_matching_point_measurement_optics()

    # from IPython import embed; embed()

    s_offset = all_twiss.iloc[0].s

    bl = latdraw.interfaces.lattice_from_ocelot(full_sequence, initial_offset=[0, 0, 23.2])
    fig, (mx, axt, axd, axe) = latdraw.subplots_with_lattice(bl, nrows=3)

    axt.plot(all_twiss.s, all_twiss.beta_x)
    # axt.plot(all_twiss.s, all_twiss.beta_y, label="$y$")
    axd.plot(all_twiss.s, all_twiss.Dx)
    axe.plot(all_twiss.s, all_twiss.E * 1e3)
    # axd.plot(all_twiss.s, all_twiss.Dy, label="y")

    gen = sim.qi52_matching_point_to_i1d_measurement_optics(dscan_conf)
    for i, (dispersion, twiss, _) in enumerate(gen):
        axd.plot(twiss.s, twiss.Dx, label=fr"$D_x\,=\,{dispersion}\,\mathrm{{m}}$")

        (line,) = axt.plot(twiss.s, twiss.beta_x)
        axe.plot(twiss.s, twiss.E*1e3)

    axd.set_ylabel(DX_LABEL_STRING)
    axd.legend()

    axt.set_ylabel(BETX_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    max_beta_x_up_to_screen = max(all_twiss.beta_x)
    mx.set_xlim(23.2, twiss.iloc[-1].s)
    axt.set_ylim(axt.get_ylim()[0], max_beta_x_up_to_screen * 1.1)
    axe.set_xlabel(Z_LABEL_STRING)
    mx.set_title("First A1 Cavity to I1D Measurement Linear Piecewise Optics")

    _save_fig_or_show(fig, outdir, "a1-to-i1d-design-optics.pdf")


def check_a1_to_i1d_design_optics_tracking(parray0, outdir):
    all_twiss, mlat = sim.a1_to_i1d_design_optics()
    s_offset = all_twiss.iloc[0].s

    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axax, axay, axe) = latdraw.subplots_with_lattice(bl, nrows=5)

    axbx.plot(all_twiss.s, all_twiss.beta_x, label="Linear Optics")
    axby.plot(all_twiss.s, all_twiss.beta_y)

    axax.plot(all_twiss.s, all_twiss.alpha_x)
    axay.plot(all_twiss.s, all_twiss.alpha_y)
    axe.plot(all_twiss.s, all_twiss.E*1e3)

    axe.set_xlabel(Z_LABEL_STRING)

    particle_twiss = sim.calculate_i1d_design_optics_from_tracking(parray0)
    axbx.plot(particle_twiss.s, particle_twiss.beta_x, label="Particle Tracking", marker="x")
    axby.plot(particle_twiss.s, particle_twiss.beta_y, marker="x")#, label="Particle Tracking")
    axax.plot(particle_twiss.s, particle_twiss.alpha_x, label="Particle Tracking", marker="x")
    axay.plot(particle_twiss.s, particle_twiss.alpha_y, marker="x")#, label="Particle Tracking")
    axe.plot(particle_twiss.s, particle_twiss.E*1e3, marker="x")#, label="Particle Tracking")
    axbx.legend()

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axax.set_ylabel(ALFX_LABEL_STRING)
    axay.set_ylabel(ALFY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    plt.show()

# def b2_dscan_optics(dscan_conf, outdir=None):
#     # fig, (ax0, ax, ax2) = plt.subplots(nrows=3, sharex=True)


#     sequence = lattice.make_to_b2d_lattice().to_magnetic_lattice().sequence
#     bl = latdraw.interfaces.lattice_from_ocelot(sequence)
#     # bl = latdraw.interfaces.lattice_from_ocelosequence())
#     bl.add_offset([0, 0, sim.S_START])
#     fig, (mx, ax0, ax, ax2) = latdraw.subplots_with_lattice(bl, nrows=3)

#     screen_s, _ = sim.get_element_s_bounds("OTRA.473.B2D")

#     # from IPython import embed; embed()

#     bc2_tds_1_start, _ = sim.get_element_s_bounds("TDSB.428.B2")

#     twisses = []

#     for i, (dispersion, twiss, _) in enumerate(sim.calculate_b2_dscan_optics(dscan_conf)):
#         twisses.append(twiss)
#         ax.plot(twiss.s, twiss.Dy, label=fr"$D_y\,=\,{dispersion}\,\mathrm{{m}}$")
#         # from IPython import embed; embed()
#         xlabel = ylabel = None
#         if i == 3:
#             xlabel = "$x$"
#             ylabel = "$y$"

#         (line,) = ax0.plot(twiss.s, twiss.beta_y, label=ylabel)
#         # colour = line.get_color()
#         # ax0.plot(twiss.s, twiss.beta_x, color=colour, linestyle="--", label=xlabel)
#         ax2.plot(twiss.s, twiss.E * 1e3)
#     ax.legend()
#     ax0.legend()


#     ax0.axhline(2.875)

#     ax0.set_xlim(bc2_tds_1_start-2.5, screen_s+2.5)
#     twisses = pd.concat(twisses)
#     twisses = twisses[(twisses.s > bc2_tds_1_start) & (twisses.s < screen_s)]

#     ax0.set_ylim(-5, max(twisses.beta_y * 1.1))
#     ax.set_ylim(-0.1, max(twisses.Dy * 1.2))



#     vlargs = {"x": screen_s, "label": "OTRA.473.B2D", "linestyle": ":", "color": "black"}
#     ax.axvline(**vlargs)
#     ax0.axvline(**vlargs)
#     ax2.axvline(**vlargs)

#     ax2.set_xlabel(r"$s$ / m")
#     ax.set_ylabel(r"$D_y$ / m")
#     ax2.set_ylabel(r"$E$ / MeV")
#     ax0.set_ylabel(r"$\beta$ / m")


#     # # ax0.set_ylim(-400, 7000)

#     if outdir is None:
#         plt.show()
#     else:
#         fname = outdir / "i1d-dscan-optics.pdf"
#         LOG.info(f"wrote {fname}")
#         fig.savefig(fname)

#     pass

# def print_optics_at_screen():
#     pass
