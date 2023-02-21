# Goal: Transport to Inj dump screen.  Do dispersion scan at 130MeV
# Goal: Transport to BC2 dump screen.  Do dispersion scan at 130MeV
#                                      Do energy scan from 2.4GeV to 700MeV

import logging
from copy import deepcopy
import pandas as pd

from ocelot.cpbd.io import load_particle_array, save_particle_array
from ocelot.cpbd.optics import twiss as oce_calc_twiss, Twiss
from ocelot.utils.fel_track import FELSimulationConfig
from ocelot.cpbd.track import track

from .sections import sections
from .sections.i1 import make_twiss0_at_cathode, Z_START
from . import lattice

LOG = logging.getLogger(__name__)

def make_i1_dscan_simulation_configs(dscan_conf, sc=True, csr=True, wake=True, smooth=True, tds_phi=0.0, tds_v=0.00042):
    dscan_magnet_configs = _dscan_conf_to_magnetic_config_dicts(dscan_conf)
    dispersions = dscan_conf.dispersions
    high_level_config = make_pre_qi52_measurement_config(sc=sc,
                                                         csr=csr,
                                                         wake=wake,
                                                         smooth=smooth,
                                                         tds_phi=tds_phi,
                                                         tds_v=tds_v)

    result = []
    for mconf, dispersion in zip(dscan_magnet_configs, dispersions):
        hlc = deepcopy(high_level_config)
        hlc.components = mconf
        hlc.metadata = {"dispersion": dispersion}
        result.append(hlc)

    return result





def make_pre_qi52_measurement_config(sc=True, csr=True, wake=True, smooth=True, tds_phi=0.0, tds_v=0.00042):
    felconf = FELSimulationConfig(physopt={"sc": sc, "csr": csr, "wake": wake, "smooth": smooth})

    felconf = FELSimulationConfig()
    felconf.a1.phi = 0
    felconf.a1.v = 125 / 8 * 1e-3

    felconf.ah1.active = False

    felconf.tds1.phi = tds_phi
    felconf.tds1.v = tds_v


    return felconf

# def make_b2_dscan_config(sc=True, csr=True, wake=True, smooth=True, tds_phi=0.0, tds_v=0.0):
#     r1 = 0.5 / 0.1366592804  # 3.6587343247857467
#     r2 = 0.5 / 0.0532325422  # 9.392750737348779
#     r3 = 0.5 / 0.0411897704  # 12.138936321917445

#     match = False

#     config = {"A1": {"phi": 0., "v": 125 / 8 * 1e-3, "SC": sc,
#                    "smooth": smooth, "wake": wake},
#               "AH1": {"phi": 0., "v": 0.,
#                     "match": False, "SC": sc, "wake": wake},
#               "LH": {"SC": sc, "CSR": csr, "wake": wake, "match": True, "tds.phi": tds_phi, "tds.v": tds_v},
#               "DL": {"SC": sc, "CSR": csr, "wake": wake, "match": False},
#               "BC0": {"rho": r1,
#                     "match": False, "SC": sc, "CSR": csr, "wake": wake},
#               "L1": {"phi": 0, "v": 0, "match": False,
#                    "SC": sc, "wake": wake, "smooth": smooth},
#               "BC1": {"rho": r2,
#                     "match": match, "SC": sc, "CSR": csr, "wake": wake},
#               "L2": {"phi": 0.0, "v": 0.0, "match": False,
#                    "SC": sc, "wake": wake, "smooth": smooth},
#               "BC2": {"rho": r3,
#                     "match": match, "SC": sc, "CSR": csr, "wake": wake},
#               "B2D": {"SC": sc, "CSR": csr, "wake": wake, "match": False}
#               }
#     return config


def make_twiss_at_q52():
    tws = Twiss()
    tws.beta_x = 3.131695851
    tws.beta_y = 5.417462794
    tws.alpha_x = -0.9249364470
    tws.alpha_y = 1.730107901
    tws.gamma_x = (1 + tws.alpha_x ** 2) / tws.beta_x
    tws.E = 0.13

    z_cathode = make_twiss0_at_cathode().s
    i1dlat = lattice.make_to_i1d_lattice()
    s_start_qi52 = i1dlat.get_element_end_s("matching-point-at-start-of-q52")

    tws.s = z_cathode + s_start_qi52

    return tws

def calculate_i1d_design_optics():
    i1dlat = lattice.make_to_i1d_lattice()
    # i1dlat.apply_high_level_config()
    twiss0 = make_twiss0_at_cathode()
    # twiss52 = make_twiss_at_q52()
    return i1dlat.calculate_twiss(twiss0=twiss0)

def run_i1_dispersion_scan(dscan_conf, fparray, dirname, fast=False):
    i1dlat = lattice.make_to_i1d_lattice()

    # First we track to the matching point.
    # Then we track 4 separate times for each dispersion scan setpoint.
    # Then we are done.
    parray0 = load_particle_array(fparray)


    a1_to_qi52_config = make_pre_qi52_measurement_config()
    navis = lattice.make_to_i1d_lattice().to_navigators(start="astra_ocelot_interface",
                                                        stop="matching-point-at-start-of-q52",
                                                        felconfig=a1_to_qi52_config)




    assert len(i1dlat.sections) == 5
    # I1D doesn't make it into the navigators:
    assert len(navis) == 4

    # Remove G1, we don't track G1, we start after it.
    # MAybe this is the wrong way to do this ...

    # we shoudl track G1 maybe I guess but it should simply return itself immediately.  this is cleaner.
    navis = navis
    sections = i1dlat.sections

    for section, navi in zip(sections, navis):
        # if sum([x.l for x in navi.lat.sequence]) == 0 and
        end = navi.lat.sequence[-1]
        LOG.info(f'Starting tracking for section "{section.name}" from {navi.lat.sequence[0]} to {end}.')
        _, parray1 = track(navi.lat, parray0, navi, overwrite_progress=False)
        fname = f"{section.name}_section_@_{end.id}.npz"
        LOG.info(f"Writing to {fname}")
        save_particle_array(fname, parray1)
        parray0 = parray1

    from IPython import embed; embed()

# def calculate_b2_dscan_optics(dscan_conf):
#     b2dlat = lattice.make_to_b2d_lattice()
#     yield from _calculate_dscan_optics(b2dlat, make_b2_dscan_config(), dscan_conf)

def _quadrupole_setting_to_config_dict(qsetting):
    result = {}
    for quad_name, k1l in zip(qsetting.names, qsetting.k1ls):
        result[quad_name] = {"k1l": k1l}
    return result

def _dscan_conf_to_magnetic_config_dicts(dscan_conf) -> dict:
    result = []
    ref_qset_config = _quadrupole_setting_to_config_dict(dscan_conf.reference_setting)
    for qsetting in dscan_conf.scan_settings:
        this_scan_setpoint_quads_config = _quadrupole_setting_to_config_dict(qsetting)
        result.append(ref_qset_config | this_scan_setpoint_quads_config)
    return result


def _calculate_dscan_optics(lat, twiss0, dscan_sim_configs, start, stop):
    s_offset = lat.get_element_end_s(start)

    for setpoint_sim_conf in dscan_sim_configs:
        twiss0 = make_twiss_at_q52()
        dispersion = setpoint_sim_conf.metadata["dispersion"]
        full_twiss, mlat = lat.calculate_twiss(twiss0, start=start, stop=stop,
                                               felconfig=setpoint_sim_conf)

        yield dispersion, full_twiss, lat


def cathode_to_first_a1_cavity_optics():
    twiss0 = sections.i1.make_twiss0_at_cathode()
    i1dlat = lattice.make_to_i1d_lattice()

    all_twiss, mlat = i1dlat["G1"].calculate_twiss(twiss0)
    return all_twiss, mlat

def a1_to_i1d_design_optics():
    gun_twiss, _ = cathode_to_first_a1_cavity_optics()

    start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
    a1_twiss0 = Twiss.from_series(gun_twiss.iloc[-1])
    i1dlat = lattice.make_to_i1d_lattice()
    all_twiss, mlat = i1dlat.calculate_twiss(a1_twiss0, start=start_name)


    return all_twiss, mlat

def a1_to_q52_matching_point_measurement_optics():
    gun_twiss, _ = cathode_to_first_a1_cavity_optics()

    start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
    # start_name = "astra_ocelot_interface_A1_section_start"
    stop_name = "matching-point-at-start-of-q52"
    a1_twiss0 = Twiss.from_series(gun_twiss.iloc[-1])

    i1dlat = lattice.make_to_i1d_lattice()
    felconfig = make_pre_qi52_measurement_config()
    return i1dlat.calculate_twiss(a1_twiss0, start=start_name, stop=stop_name, felconfig=felconfig)


def qi52_matching_point_to_i1d_measurement_optics(dscan_quad_settings):
    i1dlat = lattice.make_to_i1d_lattice()
    start = "matching-point-at-start-of-q52"
    stop = None
    twiss0 = make_twiss_at_q52()

    fel_sim_configs = make_i1_dscan_simulation_configs(dscan_quad_settings)
    yield from _calculate_dscan_optics(i1dlat, twiss0,
                                       fel_sim_configs,
                                       start=start, stop=stop)



# def qi52_matching_point_to_i1d_measurement_optics():
#     q52_twiss, _ = a1_to_q52_matching_point_measurement_optics()

#     start_name = "matching-point-at-start-of-q52"
#     stop_name = None

#     q52_twiss0 = Twiss.from_series(q52_twiss.iloc[-1])
#     i1dlat = lattice.make_to_i1d_measurement_lattice()
#     mlat = i1dlat.to_magnetic_lattice(start=start_name, stop=stop_name)

#     return q52_twiss0, i1dlat
