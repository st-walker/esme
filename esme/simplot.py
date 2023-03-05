import logging

import matplotlib.pyplot as plt
import latdraw
import pandas as pd
from ocelot.cpbd.beam import Twiss, optics_from_moments, moments_from_parray
from . import sim
from .sections import sections
from . import lattice


LOG = logging.getLogger(__name__)

MAD8 = "/Users/stuartwalker/physics/gpt-csr/mad8DesyPublic/TWISS_I1D"


Z_LABEL_STRING = r"$s$ / m"
E_LABEL_STRING = r"$E$ / MeV"
BETA_LABEL_STRING = r"$\beta$ / m"
BETX_LABEL_STRING = r"$\beta_x$ / m"
BETY_LABEL_STRING = r"$\beta_y$ / m"
ALFX_LABEL_STRING = r"$\alpha_x$"
ALFY_LABEL_STRING = r"$\alpha_y$"
DX_LABEL_STRING = "$D_x$ / m"
DY_LABEL_STRING = "$D_y$ / m"
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

    all_twiss, _ = sim.a1_to_q52_matching_point_measurement_optics()
    s_offset = all_twiss.iloc[0].s
    bl = latdraw.interfaces.lattice_from_ocelot(full_sequence, initial_offset=[0, 0, 23.2])
    fig, (mx, axt, axd, axe) = latdraw.subplots_with_lattice(bl, nrows=3)

    gen = sim.a1_dscan_piecewise_optics(dscan_conf, do_physics=do_physics)
    for i, (dispersion, twiss, _) in enumerate(gen):
        axd.plot(twiss.s, twiss.Dx, label=fr"$D_x\,=\,{dispersion}\,\mathrm{{m}}$")

        (line,) = axt.plot(twiss.s, twiss.beta_x)
        axe.plot(twiss.s, twiss.E*1e3)

    axd.set_ylabel(DX_LABEL_STRING)
    axd.legend()

    axt.set_ylabel(BETX_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    max_beta_x_up_to_screen = max(all_twiss.beta_x)
    mx.set_xlim(0, twiss.iloc[-1].s)
    axt.set_ylim(axt.get_ylim()[0], max_beta_x_up_to_screen * 1.1)
    axe.set_xlabel(Z_LABEL_STRING)
    mx.set_title("First A1 Cavity to I1D Measurement Linear Piecewise Optics")

    _save_fig_or_show(fig, outdir, "a1-to-i1d-design-optics.pdf")


def check_a1_to_i1d_design_optics_tracking(parray0, outdir):
    all_twiss, mlat = sim.a1_to_i1d_design_optics()
    s_offset = all_twiss.iloc[0].s

    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axax, axay, axdx, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=7)

    axbx.plot(all_twiss.s, all_twiss.beta_x, label="Linear Optics")
    axby.plot(all_twiss.s, all_twiss.beta_y)

    axax.plot(all_twiss.s, all_twiss.alpha_x)
    axay.plot(all_twiss.s, all_twiss.alpha_y)
    axdx.plot(all_twiss.s, all_twiss.Dx)
    axdy.plot(all_twiss.s, all_twiss.Dy)
    axe.plot(all_twiss.s, all_twiss.E*1e3)

    axe.set_xlabel(Z_LABEL_STRING)

    particle_twiss, _ = sim.i1d_design_optics_from_tracking(parray0)
    axbx.plot(particle_twiss.s, particle_twiss.beta_x, label="Particle Tracking", marker="x")
    axby.plot(particle_twiss.s, particle_twiss.beta_y, marker="x")#, label="Particle Tracking")
    axax.plot(particle_twiss.s, particle_twiss.alpha_x, label="Particle Tracking", marker="x")
    axay.plot(particle_twiss.s, particle_twiss.alpha_y, marker="x")#, label="Particle Tracking")
    axe.plot(particle_twiss.s, particle_twiss.E*1e3, marker="x")#, label="Particle Tracking")

    axdx.set_ylabel(DX_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)

    axdx.plot(particle_twiss.s, particle_twiss.Dx, marker="x")
    axdy.plot(particle_twiss.s, particle_twiss.Dy, marker="x")

    axbx.legend()

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axax.set_ylabel(ALFX_LABEL_STRING)
    axay.set_ylabel(ALFY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    plt.show()

def check_a1_q52_measurement_optics_tracking(fparray0, outdir):
    all_twiss, mlat = sim.a1_to_q52_matching_point_measurement_optics()
    s_offset = all_twiss.iloc[0].s
    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axdx, axe) = latdraw.subplots_with_lattice(bl, nrows=3)

    particle_twiss, _ = sim.a1_to_q52_matching_point_measurement_optics_from_tracking(fparray0)

    axbx.plot(all_twiss.s, all_twiss.beta_x, label="Linear Optics")
    axdx.plot(all_twiss.s, all_twiss.Dx)
    axe.plot(all_twiss.s, all_twiss.E*1e3)

    axbx.plot(particle_twiss.s, particle_twiss.beta_x, label="Particle Tracking", marker="x")
    axdx.plot(particle_twiss.s, particle_twiss.Dx)
    axe.plot(particle_twiss.s, particle_twiss.E*1e3, marker="x")#, label="Particle Tracking")


    axbx.legend()

    axbx.set_ylabel(BETX_LABEL_STRING)
    axdx.set_ylabel(DX_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    plt.show()

def dscan_piecewise_tracking_optics(fparray0, dscan_conf, outdir, do_physics=False):
    all_twiss, mlat = sim.a1_to_i1d_design_optics()
    s_offset = all_twiss.iloc[0].s
    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axdx, axe) = latdraw.subplots_with_lattice(bl, nrows=3)

    twiss_q52 = all_twiss.iloc[-1]

    # axbx.plot(all_twiss.s, all_twiss.beta_x, label="Linear Optics")
    # axdx.plot(all_twiss.s, all_twiss.Dx)
    # axe.plot(all_twiss.s, all_twiss.E*1e3)

    gen_twiss = sim.a1_dscan_piecewise_optics(dscan_conf, do_physics=do_physics)
    gen = sim.a1_dscan_piecewise_tracked_optics(fparray0, dscan_conf, do_physics=do_physics)
    # Ptwiss is twiss from particle tracking, otwiss is from direct tracking.
    for i, (dispersion, ptwiss) in enumerate(gen):
        _, otwiss = next(gen_twiss)

        blabel1 = blabel2 = ""
        if i == 0:
            blabel1 = "Linear Optics"
            blabel2 = "Particle Tracking"

        (line1,) = axbx.plot(otwiss.s, otwiss.beta_x, label=blabel1)
        (line2,) = axdx.plot(otwiss.s, otwiss.Dx, label=fr"$D_x\,=\,{dispersion}\,\mathrm{{m}}$")
        (line3,) = axe.plot(otwiss.s, otwiss.E*1e3)

        axbx.plot(ptwiss.s, ptwiss.beta_x, label=blabel2, marker="x", linestyle="", color=line1.get_color())
        axdx.plot(ptwiss.s, ptwiss.Dx, linestyle="", marker="x", color=line2.get_color())
        axe.plot(ptwiss.s, ptwiss.E*1e3, marker="x", linestyle="", color=line3.get_color())#, label="Particle Tracking")



    screen_s = all_twiss.s[all_twiss.id == "OTRC.64.I1D"].item()
    vlargs = {"x": screen_s, "linestyle": ":", "color": "black"}
    axbx.axvline(**vlargs)
    axdx.axvline(**vlargs)
    vlargs["label"] = "OTRC.64.I1D"
    axe.axvline(**vlargs)

    axe.legend()

    legend = axbx.legend()
    axdx.legend()

    axbx.set_ylabel(BETX_LABEL_STRING)
    axdx.set_ylabel(DX_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    mx.set_title(f"Particle Tracking vs Linear Optics, Collective Effects: {do_physics}")

    plt.show()


def bolko_optics_comparison(dscan_conf, tscan_voltages):
    b2 = sim.B2DSimulatedEnergySpreadMeasurement(dscan_conf, tscan_voltages)

    twiss, mlat = b2.bolko_optics()
    import pand8
    mad8 = "/Users/stuartwalker/repos/esme-xfel/esme/sections/bolko-optics.tfs"
    df8 = pand8.read(mad8)

    s_offset = twiss.iloc[0].s
    df8.SUML += s_offset
    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy) = latdraw.subplots_with_lattice(bl, nrows=3)

    axbx.plot(df8.SUML, df8.BETX, label="MAD8")
    axbx.plot(twiss.s, twiss.beta_x, label="OCELOT")

    axby.plot(df8.SUML, df8.BETY, label="MAD8")
    axby.plot(twiss.s, twiss.beta_y, label="OCELOT")

    axdy.plot(df8.SUML, df8.DY, label="MAD8")
    axdy.plot(twiss.s, twiss.Dy, label="OCELOT")

    axbx.set_xlabel(Z_LABEL_STRING)
    axbx.legend()

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)

    mx.set_title("Special Bolko Optics for BC2 TDS Operation, OCELOT and MAD8 Optics to B2D")

    tds_name = "TDSB.428.B2"

    plt.show()


def new_tds_optics_comparison(dscan_conf, tscan_voltages):
    b2 = sim.B2DSimulatedEnergySpreadMeasurement(dscan_conf, tscan_voltages)

    twiss, mlat = b2.bolko_optics()
    import pand8
    mad8 = "/Users/stuartwalker/repos/esme-xfel/esme/sections/bolko-optics.tfs"
    df8 = pand8.read(mad8)

    s_offset = 0
    bl = latdraw.interfaces.lattice_from_ocelot(mlat.sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy) = latdraw.subplots_with_lattice(bl, nrows=3)


    for dy, twiss in b2.new_optics_scan():
        # if i == 0:

        #     axbx.plot(


        axbx.plot(twiss.s, twiss.beta_x)
        axby.plot(twiss.s, twiss.beta_y)
        axdy.plot(twiss.s, twiss.Dy, label=fr"$D_y\,=\,{dy}\,\mathrm{{m}}$")

        axbx.set_xlabel(Z_LABEL_STRING)
        axdy.legend()

        axbx.set_ylabel(BETX_LABEL_STRING)
        axby.set_ylabel(BETY_LABEL_STRING)
        axdy.set_ylabel(DY_LABEL_STRING)


    screen_s = twiss.s[twiss.id == "OTRA.473.B2D"].item()
    vlargs = {"x": screen_s, "linestyle": ":", "color": "black"}
    axbx.axvline(**vlargs)
    axdy.axvline(**vlargs)
    vlargs["label"] = "OTRA.473.B2D"
    axby.axvline(**vlargs)
    axby.legend()
    axdy.set_xlabel(Z_LABEL_STRING)

    mx.set_title("Nina's B2D dispersion scan optics in OCELOT")


    plt.show()


def gun_to_b2d_bolko_optics(b2_dscan_conf, b2_tscan_voltages):
    # Plot MAD8 bolko optics

    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages)
    sequence = b2d.gun_to_dump_sequence()
    s_offset = 0

    twiss, mlat = b2d.bolko_optics()
    import pand8
    mad8 = "/Users/stuartwalker/repos/esme-xfel/esme/sections/bolko-optics.tfs"
    df8 = pand8.read(mad8)


    ll = sim.XFELLonglist("/Users/stuartwalker/repos/esme-xfel/bin/longlist.csv")
    name2_bolko_start = df8.iloc[1].NAME
    name1_bolko_start = ll.name2_to_name1(name2_bolko_start).item()

    s_offset_df8 = b2d.b2dlat.get_element_start_s(name1_bolko_start)

    df8.SUML += s_offset_df8 + s_offset

    bl = latdraw.interfaces.lattice_from_ocelot(sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy) = latdraw.subplots_with_lattice(bl, nrows=3)

    # from IPython import embed; embed()

    gtwiss, _ = b2d.gun_to_b2d_bolko_optics()

    axbx.plot(df8.SUML, df8.BETX, label="MAD Bolko")
    axbx.plot(gtwiss.s, gtwiss.beta_x, label="OCE Bolko")

    axby.plot(df8.SUML, df8.BETY, label="MAD Bolko")
    axby.plot(gtwiss.s, gtwiss.beta_y, label="OCELOT")

    axdy.plot(df8.SUML, df8.DY, label="MAD Bolko")
    axdy.plot(gtwiss.s, gtwiss.Dy, label="OCELOT")

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    # axe.set_ylabel(E_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)
    axdy.set_xlabel(Z_LABEL_STRING)
    axbx.legend()

    mx.set_title("Special high-$\\beta_x$ Bolko Optics for TDS2 in OCELOT and compared with MAD8")

    plt.show()
    # from IPython import embed; embed()

def gun_to_b2d_dispersion_scan_design_energy(b2_dscan_conf, b2_tscan_voltages):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages)
    sequence = b2d.gun_to_dump_sequence()
    s_offset = 0
    bl = latdraw.interfaces.lattice_from_ocelot(sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)

    low_energy_gen = b2d.gun_to_dump_scan_optics(design_energy=False)
    for i, (dy, full_twiss) in enumerate(b2d.gun_to_dump_scan_optics(design_energy=True)):
        _, low_e_twiss = next(low_energy_gen)

        label1 = ""
        label2 = ""
        if i == 3:
            label1 = "2.4 GeV"
            label2 = "130MeV"

        (line,) = axbx.plot(full_twiss.s, full_twiss.beta_x, label=label1)
        (line,) = axbx.plot(low_e_twiss.s, low_e_twiss.beta_x, linestyle="--", color=line.get_color(), label=label2)

        (line,) = axby.plot(full_twiss.s, full_twiss.beta_y)
        (line,) = axby.plot(low_e_twiss.s, low_e_twiss.beta_y, linestyle="--", color=line.get_color())

        axdy.plot(full_twiss.s, full_twiss.Dy, label=fr"$D_y\,=\,{dy}\,\mathrm{{m}}$")

        (line,) = axe.plot(full_twiss.s, full_twiss.E*1e3)
        (line,) = axe.plot(low_e_twiss.s, low_e_twiss.E*1e3, linestyle="--", color=line.get_color())

    screen_s = full_twiss.s[full_twiss.id == "OTRA.473.B2D"].item()
    vlargs = {"x": screen_s, "linestyle": ":", "color": "black"}
    axbx.axvline(**vlargs)
    axdy.axvline(**vlargs)
    axe.axvline(**vlargs)
    vlargs["label"] = "OTRA.473.B2D"
    axby.axvline(**vlargs)
    axby.legend()

    mx.set_title("Special high-$D_y$ B2D Optics at Design- and Low-Energy")

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    axbx.legend()
    axdy.legend(loc="center left")

    plt.show()

def gun_to_b2d_piecewise_dispersion_scan_optics(b2_dscan_conf, b2_tscan_voltages):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages)
    sequence = b2d.gun_to_dump_sequence()
    bl = latdraw.interfaces.lattice_from_ocelot(sequence)
    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)

    low_energy_gen = b2d.gun_to_dump_piecewise_scan_optics()
    for i, (dy, full_twiss) in enumerate(b2d.gun_to_dump_scan_optics(design_energy=True)):
        _, low_e_twiss, matching_points = next(low_energy_gen)

        label1 = ""
        label2 = ""
        if i == 3:
            label1 = "2.4 GeV"
            label2 = "130 MeV"

        (line,) = axbx.plot(full_twiss.s, full_twiss.beta_x, label=label1)
        (line,) = axbx.plot(low_e_twiss.s, low_e_twiss.beta_x, linestyle="--", color=line.get_color(), label=label2)

        (line,) = axby.plot(full_twiss.s, full_twiss.beta_y)
        (line,) = axby.plot(low_e_twiss.s, low_e_twiss.beta_y, linestyle="--", color=line.get_color())

        axdy.plot(full_twiss.s, full_twiss.Dy, label=fr"$D_y\,=\,{dy}\,\mathrm{{m}}$")

        (line,) = axe.plot(full_twiss.s, full_twiss.E * 1e3)
        (line,) = axe.plot(low_e_twiss.s, low_e_twiss.E * 1e3, linestyle="--", color=line.get_color())

    for i, matching_point in enumerate(matching_points):
        s = b2d.b2dlat.get_element_end_s(matching_point)
        axbx.axvline(s, linestyle="-.", color="green", alpha=0.7)

        if i == 0:
            axby.axvline(s, linestyle="-.", color="green", alpha=0.7, label="Matching Points")
        axby.axvline(s, linestyle="-.", color="green", alpha=0.7)
        axdy.axvline(s, linestyle="-.", color="green", alpha=0.7)
        axe.axvline(s, linestyle="-.", color="green", alpha=0.7)


    screen_s = b2d.b2dlat.get_element_end_s("OTRA.473.B2D")
    vlargs = {"x": screen_s, "linestyle": ":", "color": "black"}
    axbx.axvline(**vlargs)
    axdy.axvline(**vlargs)
    axe.axvline(**vlargs)
    vlargs["label"] = "OTRA.473.B2D"
    axby.axvline(**vlargs)
    axby.legend()


    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    axbx.legend()
    axdy.legend(loc="center left")

    plt.show()


def gun_to_b2d_tracking_piecewise_optics(b2_dscan_conf, b2_tscan_voltages, fparray0):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages, fparray0=fparray0)
    sequence = b2d.gun_to_dump_sequence()
    s_offset = 0
    bl = latdraw.interfaces.lattice_from_ocelot(sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)

    # Particle Tracking optics
    low_energy_gen = b2d.gun_to_dump_piecewise_scan_optics_tracking()
    # Some twiss optics of some sort:
    for i, (dy, full_twiss) in enumerate(b2d.gun_to_dump_scan_optics(design_energy=True)):
        _, low_e_twiss, matching_points = next(low_energy_gen)
        # matching_points = []
        # low_e_twiss = full_twiss
        label1 = ""
        label2 = ""
        if i == 3:
            label1 = "2.4 GeV Linear Optics"
            label2 = "130 MeV Tracking"

        (line,) = axbx.plot(full_twiss.s, full_twiss.beta_x, label=label1)
        (line,) = axbx.plot(low_e_twiss.s, low_e_twiss.beta_x, linestyle="--", color=line.get_color(), label=label2)

        (line,) = axby.plot(full_twiss.s, full_twiss.beta_y)
        (line,) = axby.plot(low_e_twiss.s, low_e_twiss.beta_y, linestyle="--", color=line.get_color())

        axdy.plot(full_twiss.s, full_twiss.Dy, label=fr"$D_y\,=\,{dy}\,\mathrm{{m}}$")

        (line,) = axe.plot(full_twiss.s, full_twiss.E * 1e3)
        (line,) = axe.plot(low_e_twiss.s, low_e_twiss.E * 1e3, linestyle="--", color=line.get_color())

    for i, matching_point in enumerate(matching_points):
        s = b2d.b2dlat.get_element_end_s(matching_point)
        axbx.axvline(s, linestyle="-.", color="green", alpha=0.7)

        if i == 0:
            axby.axvline(s, linestyle="-.", color="green", alpha=0.7, label="Matching Points")
        axby.axvline(s, linestyle="-.", color="green", alpha=0.7)
        axdy.axvline(s, linestyle="-.", color="green", alpha=0.7)
        axe.axvline(s, linestyle="-.", color="green", alpha=0.7)


    screen_s = b2d.b2dlat.get_element_end_s("OTRA.473.B2D")
    vlargs = {"x": screen_s, "linestyle": ":", "color": "black"}
    axbx.axvline(**vlargs)
    axdy.axvline(**vlargs)
    axe.axvline(**vlargs)
    vlargs["label"] = "OTRA.473.B2D"
    axby.axvline(**vlargs)
    axby.legend()

    mx.set_title("B2D Dispersion Scan Optics at design eneryg and 130MeV with artificial matching")


    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    axbx.legend()
    axdy.legend(loc="center left")

    plt.show()

def gun_to_b2d_tracking_central_slice_optics(b2_dscan_conf, b2_tscan_voltages, fparray0, do_physics=False, outdir=None):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages, fparray0=fparray0)
    sequence = b2d.gun_to_dump_sequence()
    s_offset = 0
    bl = latdraw.interfaces.lattice_from_ocelot(sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)

    # Particle Tracking optics that we are trying to make nice.
    low_energy_gen = b2d.gun_to_dump_central_slice_optics(do_physics=do_physics, outdir=outdir)

    # The design optis that we are aiming for:
    for i, (dy, full_twiss) in enumerate(b2d.gun_to_dump_scan_optics(design_energy=True)):
        _, low_e_twiss, matching_points = next(low_energy_gen)
        # matching_points = []
        # low_e_twiss = full_twiss
        label1 = ""
        label2 = ""
        if i == 3:
            label1 = "2.4 GeV Linear Optics"
            label2 = "130 MeV Tracking"

        # from IPython import embed; embed()
        marker = ""

        (line,) = axbx.plot(full_twiss.s, full_twiss.beta_x, label=label1)
        (line,) = axbx.plot(low_e_twiss.s, low_e_twiss.beta_x, linestyle="--", color=line.get_color(), label=label2, marker=marker)

        (line,) = axby.plot(full_twiss.s, full_twiss.beta_y)
        (line,) = axby.plot(low_e_twiss.s, low_e_twiss.beta_y, linestyle="--", color=line.get_color(), marker=marker)

        axdy.plot(full_twiss.s, full_twiss.Dy, label=fr"$D_y\,=\,{dy}\,\mathrm{{m}}$")

        (line,) = axe.plot(full_twiss.s, full_twiss.E * 1e3)
        (line,) = axe.plot(low_e_twiss.s, low_e_twiss.E * 1e3, linestyle="--", color=line.get_color(), marker=marker)

    for i, matching_point in enumerate(matching_points):
        s = b2d.b2dlat.get_element_end_s(matching_point)
        axbx.axvline(s, linestyle="-.", color="green", alpha=0.7)

        if i == 0:
            axby.axvline(s, linestyle="-.", color="green", alpha=0.7, label="Matching Points")
        axby.axvline(s, linestyle="-.", color="green", alpha=0.7)
        axdy.axvline(s, linestyle="-.", color="green", alpha=0.7)
        axe.axvline(s, linestyle="-.", color="green", alpha=0.7)


    screen_s = b2d.b2dlat.get_element_end_s("OTRA.473.B2D")
    vlargs = {"x": screen_s, "linestyle": ":", "color": "black"}
    axbx.axvline(**vlargs)
    axdy.axvline(**vlargs)
    axe.axvline(**vlargs)
    vlargs["label"] = "OTRA.473.B2D"
    axby.axvline(**vlargs)
    axby.legend()

    mx.set_title("B2D Dispersion Scan Optics at Design Energy and 130MeV "
                 f"with Artificially Matched Central Slices: Physics: {do_physics}")


    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    axbx.legend()
    axdy.legend(loc="center left")

    plt.show()



def gun_to_b2d_dispersion_scan_low_energy(b2_dscan_conf, b2_tscan_voltages):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages)
    sequence = b2d.gun_to_dump_sequence()
    s_offset = 0.0
    bl = latdraw.interfaces.lattice_from_ocelot(sequence, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)

    for dy, full_twiss in b2d.gun_to_dump_scan_optics(design_energy=False):

        axbx.plot(full_twiss.s, full_twiss.beta_x)
        axby.plot(full_twiss.s, full_twiss.beta_y)
        axdy.plot(full_twiss.s, full_twiss.Dy, label=fr"$D_y\,=\,{dy}\,\mathrm{{m}}$")
        axe.plot(full_twiss.s, full_twiss.E*1e3)

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axe.set_ylabel(E_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    axdy.legend(loc="center left")

    plt.show()



def plot_b2d_design_optics(b2_dscan_conf, b2_tscan_voltages):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages)

    sequence = b2d.gun_to_dump_sequence()
    s_offset = 0


    import pandas as pd
    igor_lat = pd.read_pickle("/Users/stuartwalker/repos/esme-xfel/bin/igors-b2d-lattice.pcl")
    bl_igor = latdraw.interfaces.lattice_from_ocelot(igor_lat, initial_offset=[0, 0, 3.2])
    bl = latdraw.interfaces.lattice_from_ocelot(sequence, initial_offset=[0, 0, 0])


    bl8 = latdraw.interfaces.read_mad8("/Users/stuartwalker/repos/esme-xfel/esme/sections/TWISS_B2D")
    # from IPython import embed; embed()
    bl8.add_offset([0, 0, 0])


    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)
    # fig, (mx, mx2, axbx, axby, axdy, axax, axay) = latdraw.subplots_with_lattices([bl, bl_igor, None, None, None, None, None])

    import pandas as pd
    itwiss = pd.read_pickle(b2d.IGOR_BC2)

    itwiss.s += 3.2

    # import pand8
    otwiss, _ = b2d.design_optics()
    # mtwiss = pand8.read(b2d.B2D_DESIGN_OPTICS)
    # mtwiss.SUML += 23.2

    axbx.plot(otwiss.s, otwiss.beta_x, label="ESME OCELOT", marker="x")
    axbx.plot(itwiss.s, itwiss.beta_x, label="Reference OCELOT", marker="x")

    axby.plot(otwiss.s, otwiss.alpha_x, marker="x")
    axby.plot(itwiss.s, itwiss.alpha_x, marker="x")


    axdy.plot(otwiss.s, otwiss.Dy)
    axdy.plot(itwiss.s, itwiss.Dy)

    axe.plot(otwiss.s, otwiss.E)
    axe.plot(itwiss.s, itwiss.E)


    axbx.set_ylim(-5, 100)
    axby.set_ylim(-5, 100)

    axbx.set_ylabel(BETX_LABEL_STRING)
    axby.set_ylabel(BETY_LABEL_STRING)
    axdy.set_ylabel(DY_LABEL_STRING)

    axe.set_ylabel(E_LABEL_STRING)


    axe.set_xlabel(Z_LABEL_STRING)

    axbx.legend()

    plt.show()

    # from IPython import embed; embed()


def piecewise_a1_to_b2d_optics(b2_dscan_conf, b2_tscan_voltages):
    b2d = sim.B2DSimulatedEnergySpreadMeasurement(b2_dscan_conf, b2_tscan_voltages)


    seq = b2d.b2dlat.to_navigator().lat.sequence
    s_offset = 0
    bl = latdraw.interfaces.lattice_from_ocelot(seq, initial_offset=[0, 0, s_offset])
    fig, (mx, axbx, axby, axdy, axe) = latdraw.subplots_with_lattice(bl, nrows=4)



    import pand8
    mad8 = "/Users/stuartwalker/repos/esme-xfel/esme/sections/bolko-optics.tfs"
    df8 = pand8.read(mad8)
    ll = sim.XFELLonglist("/Users/stuartwalker/repos/esme-xfel/bin/longlist.csv")
    name1 = ll.name2_to_name1(df8.iloc[1].NAME)



    for dy, full_twiss in b2d.full_scan_optics():

        mad8_offset = full_twiss[full_twiss.id == name1].s

        # from IPython import embed; embed()

        axbx.plot(full_twiss.s, full_twiss.beta_x)
        axe.plot(full_twiss.s, full_twiss.E*1e3)


    axe.set_ylabel(E_LABEL_STRING)
    axbx.set_ylabel(BETX_LABEL_STRING)
    axe.set_xlabel(Z_LABEL_STRING)

    axe.legend()

    plt.show()
