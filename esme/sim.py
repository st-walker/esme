# Goal: Transport to Inj dump screen.  Do dispersion scan at 130MeV
# Goal: Transport to BC2 dump screen.  Do dispersion scan at 130MeV
#                                      Do energy scan from 2.4GeV to 700MeV

import logging
from copy import deepcopy

from ocelot.cpbd.io import load_particle_array
from ocelot.cpbd.optics import twiss as oce_calc_twiss, Twiss
from ocelot.utils.fel_track import FELSimulationConfig

from .sections import sections
from .sections.i1 import make_twiss0_at_cathode, Z_START
from . import lattice

LOG = logging.getLogger(__name__)

LoadRF = "RF_250_5_2M.txt"  # RF parameters
def make_i1_dscan_simulation_configs(dscan_conf, sc=True, csr=True, wake=True, smooth=True, tds_phi=0.0, tds_v=0.00042):
    # c=299792458; grad=np.pi/180; f = 1.3e9; k=2*np.pi*f/c


    dscan_magnet_configs = _dscan_conf_to_magnetic_config_dicts(dscan_conf)
    dispersions = dscan_conf.dispersions

    high_level_config = {"A1": {"phi": 0., "v": 125 / 8 * 1e-3, "SC": sc,
                   "smooth": smooth, "wake": wake},
              "AH1": {"phi": 0., "v": 0.,
                    "match": False, "SC": sc, "wake": wake},
              "LH": {"SC": sc, "CSR": csr, "wake": wake, "match": True, "tds.phi": tds_phi, "tds.v": tds_v},
              "I1D": {"SC": sc, "CSR": csr, "wake": wake, "match": False}
              }

    result = []
    for mconf, dispersion in zip(dscan_magnet_configs, dispersions):
        result.append(FELSimulationConfig(sections=deepcopy(high_level_config),
                                          magnets=mconf, metadata={"dispersion": dispersion}))

    return result


def make_b2_dscan_config(sc=True, csr=True, wake=True, smooth=True, tds_phi=0.0, tds_v=0.0):
    r1 = 0.5 / 0.1366592804  # 3.6587343247857467
    r2 = 0.5 / 0.0532325422  # 9.392750737348779
    r3 = 0.5 / 0.0411897704  # 12.138936321917445

    match = False

    config = {"A1": {"phi": 0., "v": 125 / 8 * 1e-3, "SC": sc,
                   "smooth": smooth, "wake": wake},
              "AH1": {"phi": 0., "v": 0.,
                    "match": False, "SC": sc, "wake": wake},
              "LH": {"SC": sc, "CSR": csr, "wake": wake, "match": True, "tds.phi": tds_phi, "tds.v": tds_v},
              "DL": {"SC": sc, "CSR": csr, "wake": wake, "match": False},
              "BC0": {"rho": r1,
                    "match": False, "SC": sc, "CSR": csr, "wake": wake},
              "L1": {"phi": 0, "v": 0, "match": False,
                   "SC": sc, "wake": wake, "smooth": smooth},
              "BC1": {"rho": r2,
                    "match": match, "SC": sc, "CSR": csr, "wake": wake},
              "L2": {"phi": 0.0, "v": 0.0, "match": False,
                   "SC": sc, "wake": wake, "smooth": smooth},
              "BC2": {"rho": r3,
                    "match": match, "SC": sc, "CSR": csr, "wake": wake},
              "B2D": {"SC": sc, "CSR": csr, "wake": wake, "match": False}
              }
    return config


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

# def run_i1_dispersion_scan(dscan_conf, fparray, dirname, fast=False):
#     i1dlat = lattice.make_to_i1d_lattice()

#     parray0 = load_particle_array(fparray)
#     config = make_i1_dscan_config(sc=not fast, csr=not fast, wake=not fast, smooth=not fast)
#     parray1 = i1dlat.track(parray0, config=config)

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
    # lat.apply_high_level_config(config)

    # lattice.apply_quad_setting_to_lattice(lat, dscan_conf.reference_setting)

    # reference_qconfig = _quadrupole_setting_to_config_dict(dscan_conf.reference_setting)

    s_offset = lat.get_element_end_s(start)

    for setpoint_sim_conf in dscan_sim_configs:
        # lattice.apply_quad_setting_to_lattice(lat, qsetting)
        mlat = lat.to_magnetic_lattice(start=start, stop=stop, felconfig=setpoint_sim_conf)
        twiss0 = make_twiss_at_q52()
        # full_twiss = lat.calculate_twiss(twiss0=twiss0, config=mag_config)
        dispersion = setpoint_sim_conf.metadata["dispersion"]

        full_twiss = oce_calc_twiss(mlat, twiss0, return_df=True)

        # full_twiss.s += s_offset

        yield dispersion, full_twiss, lat



def cathode_to_first_a1_cavity_optics():
    twiss0 = sections.i1.make_twiss0_at_cathode()
    i1dlat = lattice.make_to_i1d_measurement_lattice()

    mlat = i1dlat.to_magnetic_lattice(start=None, stop="astra_ocelot_interface")

    return oce_calc_twiss(mlat, twiss0, return_df=True), mlat

def a1_to_i1d_design_optics():
    gun_twiss, _ = cathode_to_first_a1_cavity_optics()


    start_name = "astra_ocelot_interface"
    a1_twiss0 = Twiss.from_series(gun_twiss.iloc[-1])
    i1dlat = lattice.make_to_i1d_lattice()
    mlat = i1dlat.to_magnetic_lattice(start=start_name)

    all_twiss = oce_calc_twiss(mlat, a1_twiss0, return_df=True)

    return all_twiss, mlat

def a1_to_q52_matching_point_measurement_optics():
    gun_twiss, _ = cathode_to_first_a1_cavity_optics()

    start_name = "astra_ocelot_interface"
    stop_name = "matching-point-at-start-of-q52"
    a1_twiss0 = Twiss.from_series(gun_twiss.iloc[-1])
    i1dlat = lattice.make_to_i1d_measurement_lattice()
    mlat = i1dlat.to_magnetic_lattice(start=start_name, stop=stop_name)

    all_twiss = oce_calc_twiss(mlat, a1_twiss0, return_df=True)


    return all_twiss, mlat

def qi52_matching_point_to_i1d_measurement_optics(dscan_quad_settings):
    i1dlat = lattice.make_to_i1d_lattice()
    start = "matching-point-at-start-of-q52"
    stop = None
    # mlat = i1dlat.to_magnetic_lattice(start=start, stop=stop)
    twiss0 = make_twiss_at_q52()

    fel_sim_configs = make_i1_dscan_simulation_configs(dscan_quad_settings)
    yield from _calculate_dscan_optics(i1dlat, twiss0,
                                       fel_sim_configs, start=start, stop=stop)


# def qi52_matching_point_to_i1d_measurement_optics():
#     q52_twiss, _ = a1_to_q52_matching_point_measurement_optics()

#     start_name = "matching-point-at-start-of-q52"
#     stop_name = None

#     q52_twiss0 = Twiss.from_series(q52_twiss.iloc[-1])
#     i1dlat = lattice.make_to_i1d_measurement_lattice()
#     mlat = i1dlat.to_magnetic_lattice(start=start_name, stop=stop_name)

#     return q52_twiss0, i1dlat
