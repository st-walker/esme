# Goal: Transport to Inj dump screen.  Do dispersion scan at 130MeV
# Goal: Transport to BC2 dump screen.  Do dispersion scan at 130MeV
#                                      Do energy scan from 2.4GeV to 700MeV

import logging
from copy import deepcopy
from pathlib import Path

import pand8
import pandas as pd
from ocelot.cpbd.beam import beam_matching, moments_from_parray, optics_from_moments
from ocelot.cpbd.io import load_particle_array, save_particle_array
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.optics import Twiss
from ocelot.utils.fel_track import (
    EuXFELController,
    FELSimulationConfig,
    NoTwiss,
    SliceTwissCalculator,
    TwissCalculator,
)

from . import lattice
from .sections.i1 import make_twiss0_at_a1_start

LOG = logging.getLogger(__name__)
LONGLIST = "/Users/stuartwalker/repos/esme-xfel/bin/longlist.csv"


def make_i1_dscan_simulation_configs(dscan_conf, do_physics=True):
    dscan_magnet_configs = _dscan_conf_to_magnetic_config_dicts(dscan_conf)
    dispersions = dscan_conf.dispersions
    high_level_config = make_pre_qi52_measurement_config(do_physics=do_physics)

    result = []
    for mconf, dispersion in zip(dscan_magnet_configs, dispersions):
        hlc = deepcopy(high_level_config)
        hlc.controller.components = mconf
        hlc.metadata = {"dispersion": dispersion, "voltage": 0.61}

        hlc.controller.tds1.phi = 0
        # 0.61 MV (units in ocelot are in GV)
        hlc.controller.tds1.v = 0.61 * 1e-3

        result.append(hlc)

    return result


def make_i1_tscan_simulation_configs(
    reference_quad_setting, tscan_voltages, do_physics=True
):
    quad_dict = _quadrupole_setting_to_config_dict(reference_quad_setting)
    high_level_config = make_pre_qi52_measurement_config(do_physics=do_physics)

    result = []
    for voltage in tscan_voltages:
        hlc = deepcopy(high_level_config)
        hlc.controller.components = quad_dict
        hlc.metadata = {
            "dispersion": reference_quad_setting.dispersion,
            "voltage": voltage,
        }

        hlc.tds1.phi = 0
        # 0.61 MV (units in ocelot are in GV)
        hlc.tds1.v = 0.61 * 1e-3

        result.append(hlc)

    return result


def make_pre_qi52_measurement_config(do_physics=False):
    felconf = FELSimulationConfig()
    felconf.do_physics = do_physics
    # go on crest in a1.  increase voltage in a1.  disable ah1.
    felconf.controller.a1.phi = 0
    felconf.controller.a1.v = 125 / 8 * 1e-3
    felconf.controller.ah1.active = False

    return felconf


def make_twiss_at_q52():
    tws = Twiss()
    # From Sergey's old script...  Just tracking using the nominal
    # optics from before A1 will NOT work and there will be a mismatch
    # with wrong optics at teh screen!
    tws.beta_x = 3.131695851
    tws.beta_y = 5.417462794
    tws.alpha_x = -0.9249364470
    tws.alpha_y = 1.730107901
    tws.gamma_x = (1 + tws.alpha_x**2) / tws.beta_x
    tws.E = 0.13

    i1dlat = lattice.make_to_i1d_lattice(make_twiss0_at_a1_start())
    tws.s = i1dlat.get_element_end_s("matching-point-at-start-of-q52")

    return tws


def calculate_i1d_design_optics():
    i1dlat = lattice.make_to_i1d_lattice()
    # i1dlat.apply_high_level_config()
    twiss0 = make_twiss0_at_cathode()
    # twiss52 = make_twiss_at_q52()
    return i1dlat.calculate_twiss(twiss0=twiss0)


def run_i1_dispersion_scan(dscan_conf, fparray, dirname, fast=False):
    i1dlat = lattice.make_to_i1d_lattice()

    # Track to just in front of QI52
    (
        twiss_to_q52,
        parray_q52,
    ) = a1_to_q52_matching_point_measurement_optics_from_tracking(
        fparray0, do_physics=do_physics
    )

    i1dlat = lattice.make_to_i1d_lattice()
    start = "matching-point-at-start-of-q52"
    stop = None
    make_twiss_at_q52()

    fel_sim_configs = make_i1_dscan_simulation_configs(
        dscan_conf, do_physics=do_physics
    )
    for dispersion, twiss_from_q52, mlat_from_q52 in _calculate_dscan_tracked_optics(
        i1dlat, parray_q52.copy(), fel_sim_configs, start=start, stop=stop
    ):
        piecewise_twiss = pd.concat([twiss_to_q52, twiss_from_q52])
        yield dispersion, piecewise_twiss


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
    lat.get_element_end_s(start)

    for setpoint_sim_conf in dscan_sim_configs:
        twiss0 = make_twiss_at_q52()
        dispersion = setpoint_sim_conf.metadata["dispersion"]
        full_twiss, mlat = lat.calculate_twiss(
            twiss0=twiss0, start=start, stop=stop, felconfig=setpoint_sim_conf
        )

        yield dispersion, full_twiss, mlat


def _calculate_dscan_tracked_optics(lat, parray0, dscan_sim_configs, start, stop):
    lat.get_element_end_s(start)

    for setpoint_sim_conf in dscan_sim_configs:
        twiss0 = make_twiss_at_q52()
        dispersion = setpoint_sim_conf.metadata["dispersion"]
        # parray0 is automatically matched to twiss0 within track_gaussian_optics.
        full_twiss, mlat = lat.track_gaussian_optics(
            twiss0,
            start=start,
            parray0=parray0.copy(),
            stop=stop,
            felconfig=setpoint_sim_conf,
        )
        yield dispersion, full_twiss, mlat


def _run_dispersion_scan_from_matching_point(
    lat, parray_mp, dscan_sim_configs, start, stop
):
    lat.get_element_end_s(start)

    for setpoint_sim_conf in dscan_sim_configs:
        twiss0 = make_twiss_at_q52()
        dispersion = setpoint_sim_conf.metadata["dispersion"]
        # parray_mp is automatically matched to twiss0 within track_gaussian_optics.
        full_twiss, mlat = lat.track_gaussian_optics(
            twiss0,
            start=start,
            parray_mp=parray_mp.copy(),
            stop=stop,
            felconfig=setpoint_sim_conf,
        )
        yield dispersion, full_twiss, mlat


class SimulatedEnergySpreadMeasurement:
    def __init__(self, dscan_conf, tscan_voltages, fparray0):
        self.parray0 = self._load_parray0(fparray0)

        self.dscan_conf = dscan_conf
        self.tscan_voltages = tscan_voltages

    def track_to_matching_point(self, dumps=None, optics=False, physics=False):
        conf = self._low_energy_config()
        # conf = FELSimulationConfig()
        conf.do_physics = physics
        conf.matching_points = self.MATCHING_POINTS
        kwargs = {
            "parray0": self.parray0.copy(),
            "start": "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity",
            "stop": self.SCAN_MATCHING_POINT,
            "felconfig": conf,
            "design_config": FELSimulationConfig(),
        }
        if optics:
            kwargs["opticscls"] = TwissCalculator
        else:
            kwargs["opticscls"] = NoTwiss

        navi = self.dlat.to_navigator(
            start=kwargs["start"], stop=kwargs["stop"], felconfig=kwargs["felconfig"]
        )
        parray1, twiss = self.dlat.track_optics(**kwargs)
        return parray1, twiss, navi.lat.sequence

    def design_linear_optics(self):
        return self.dlat.calculate_twiss()

    def _low_energy_config(self, physics=False):
        conf = FELSimulationConfig()
        conf.controller.a2.active = False
        conf.controller.ah1.active = False
        conf.controller.a3.active = False
        conf.controller.a1.phi = 0
        conf.controller.a1.v = 123.5 / 8 * 1e-3
        # if physics:
        #     # [-1.61701102,  1.75132038, -0.68409032,  0.98491115, -0.28617504]
        #     conf.controller.components = {'Q.37.I1': {'k1': -1.61701102},
        #                                   'Q.38.I1': {'k1': 1.75132038},
        #                                   'QI.46.I1': {'k1': -0.68409032},
        #                                   'QI.47.I1': {'k1': 0.98491115},
        #                                   'QI.50.I1': {'k1': -0.28617504}}

        return conf

    def _full_dscan_configs(self, design_energy=False):
        bolko_config = self._make_bolko_optics_config()
        scan_quads = _dscan_conf_to_magnetic_config_dicts(self.dscan_conf)

        for scan_quads in _dscan_conf_to_magnetic_config_dicts(self.dscan_conf):
            nina_conf = deepcopy(bolko_config)
            if not design_energy:
                nina_conf.controller.a2.active = False
                nina_conf.controller.ah1.active = False
                nina_conf.controller.a3.active = False
                nina_conf.controller.a1.phi = 0
                nina_conf.controller.a1.v = 125 / 8 * 1e-3

            for quad_name, attrd in scan_quads.items():
                # Use bolko optics config as a baseline for nina's new config.
                nina_conf.controller.components[quad_name] = attrd

            yield nina_conf

    def matching_point_to_screen(self):
        pass

    def run_tds_scan(self, parray0):
        pass

    def run_dscan(self, parray0):
        pass

    def _load_parray0(self, fparray0):
        parray0 = load_particle_array(fparray0)
        twiss0 = self.dlat.twiss0  # This is already at the end of the gun!
        twiss_at_end_of_gun = twiss0
        LOG.info("Loading parray and matching it to twiss at end of the gun")
        self._match_parray(parray0, twiss_at_end_of_gun)
        return parray0

    @staticmethod
    def _match_parray(parray0, twiss, matching_slice=None):
        return
        beam_matching(
            parray0,
            bounds=[-5, 5],
            x_opt=[twiss.alpha_x, twiss.beta_x, 0],
            y_opt=[twiss.alpha_y, twiss.beta_y, 0],
            slice=matching_slice,
        )


class SimulatedB2DEnergySpreadMeasurement(SimulatedEnergySpreadMeasurement):
    MATCHING_POINTS = ["matching-point-at-start-of-q52", "MATCH.174.L1"]
    SCAN_MATCHING_POINT = "MATCH.428.B2"
    ESCAN_MATCHING_POINT = ""
    SCREEN_NAME = "OTRA.473.B2D"

    def __init__(self, *args, **kwargs):
        self.dlat = lattice.make_to_b2d_lattice(make_twiss0_at_a1_start())
        super().__init__(*args, **kwargs)


class SimulatedI1DEnergySpreadMeasurement(SimulatedEnergySpreadMeasurement):
    MATCHING_POINTS = None
    SCAN_MATCHING_POINT = "matching-point-at-start-of-q52"

    def __init__(self, *args, **kwargs):
        self.dlat = lattice.make_to_i1d_lattice(make_twiss0_at_a1_start())
        super().__init__(*args, **kwargs)

    def track_to_q52(self):
        if self._parrayq52 is not None:
            return self._parrayq52

        start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
        stop_name = "matching-point-at-start-of-q52"
        conf = FELSimulationConfig(do_physics=True)
        conf.ah1.active = False
        conf.a1.phi = 0
        conf.a1.v = 125 / 8 * 1e-3
        parrayq52 = self.i1dlat.track(
            parray0=self.parray0.copy(),
            start=start_name,
            stop=stop_name,
            felconfig=conf,
        )
        self._match_parray(parrayq52, self.twiss_at_q52, matching_slice="Emax")
        self._parrayq52 = parrayq52
        return self._parrayq52

    @staticmethod
    def _match_parray(parray0, twiss, matching_slice=None):
        beam_matching(
            parray0,
            bounds=[-5, 5],
            x_opt=[twiss.alpha_x, twiss.beta_x, 0],
            y_opt=[twiss.alpha_y, twiss.beta_y, 0],
            slice=matching_slice,
        )

    def run_dispersion_scan(self):
        fel_sim_configs = make_i1_dscan_simulation_configs(
            self.dscan_conf, do_physics=True
        )
        for config in fel_sim_configs:
            dispersion = config.metadata["dispersion"]
            parray_otrc64 = self.i1dlat.track(
                parray0=self.track_to_q52().copy(),
                start="matching-point-at-start-of-q52",
                stop="OTRC.64.I1D",
                felconfig=config,
            )
            yield dispersion, parray_otrc64

    def run_tds_scan(self):
        fel_sim_configs = make_i1_tscan_simulation_configs(
            self.dscan_conf.reference_setting, self.tscan_voltages, do_physics=True
        )
        for config in fel_sim_configs:
            voltage = config.metadata["voltage"]
            parray_otrc64 = self.i1dlat.track(
                parray0=self.track_to_q52().copy(),
                start="matching-point-at-start-of-q52",
                stop="OTRC.64.I1D",
                felconfig=config,
            )
            yield voltage, parray_otrc64

    def write_scans(self, outdir):
        outdir = Path(outdir)
        LOG.info("Running dispersion scan")
        dscan = self.run_dispersion_scan()
        for dispersion, parray_otrc64 in dscan:
            outpath = outdir / f"OTRC.64.I1D-dispersion-{dispersion}.npz"
            save_particle_array(outpath, parray_otrc64)
            LOG.info(f"Written: {outpath}")

        LOG.info("Running dispersion scan")
        for tds_voltage, parray_otrc64 in self.run_tds_scan():
            outpath = outdir / f"OTRC.64.I1D-voltage-{tds_voltage}.npz"
            save_particle_array(outpath, parray_otrc64)
            LOG.info(f"Written: {outpath}")


class B2DSimulatedEnergySpreadMeasurement:
    # B2D_DESIGN_OPTICS = "/Users/stuartwalker/repos/esme-xfel/esme/sections/TWISS_B2D"
    IGOR_BC2 = "/Users/stuartwalker/repos/esme-xfel/esme/sections/igor-bc2.pcl"
    BOLKO_OPTICS = "/Users/stuartwalker/repos/esme-xfel/esme/sections/bolko-optics.tfs"

    def __init__(self, dscan_conf, tscan_voltages, fparray0=None):
        self.longlist = XFELLonglist(LONGLIST)
        self.b2dlat = lattice.make_to_b2d_lattice(make_twiss0_at_a1_start())

        if fparray0 is not None:
            self.parray0 = self._load_parray0(fparray0)

        self.dscan_conf = dscan_conf
        self.tscan_voltages = tscan_voltages

    def _load_parray0(self, fparray0):
        parray0 = load_particle_array(fparray0)
        twiss0 = self.b2dlat.twiss0  # This is already at the end of the gun!
        # gun_twiss, mlat = self.b2dlat["G1"].calculate_twiss(twiss0)
        # twiss_at_end_of_gun = Twiss.from_series(gun_twiss.iloc[-1])
        twiss_at_end_of_gun = twiss0
        LOG.info("Loading parray and matching it to twiss at end of the gun")
        self._match_parray(parray0, twiss_at_end_of_gun)
        return parray0

    @staticmethod
    def _match_parray(parray0, twiss, matching_slice=None):
        beam_matching(
            parray0,
            bounds=[-5, 5],
            x_opt=[twiss.alpha_x, twiss.beta_x, 0],
            y_opt=[twiss.alpha_y, twiss.beta_y, 0],
            slice=matching_slice,
        )

    def run_dispersion_scan(self):
        fel_sim_configs = make_i1_dscan_simulation_configs(
            self.dscan_conf, do_physics=True
        )
        for config in fel_sim_configs:
            dispersion = config.metadata["dispersion"]
            parray_otrc64 = self.i1dlat.track(
                parray0=self.track_to_q52().copy(),
                start="matching-point-at-start-of-q52",
                stop="OTRC.64.I1D",
                felconfig=config,
            )
            yield dispersion, parray_otrc64

    def run_tds_scan(self):
        fel_sim_configs = make_i1_tscan_simulation_configs(
            self.dscan_conf.reference_setting, self.tscan_voltages, do_physics=True
        )
        for config in fel_sim_configs:
            voltage = config.metadata["voltage"]
            parray_otrc64 = self.i1dlat.track(
                parray0=self.track_to_q52().copy(),
                start="matching-point-at-start-of-q52",
                stop="OTRC.64.I1D",
                felconfig=config,
            )
            yield voltage, parray_otrc64

    def write_scans(self, outdir):
        outdir = Path(outdir)
        LOG.info("Running dispersion scan")
        dscan = self.run_dispersion_scan()
        for dispersion, parray_otrc64 in dscan:
            outpath = outdir / f"OTRC.64.I1D-dispersion-{dispersion}.npz"
            save_particle_array(outpath, parray_otrc64)
            LOG.info(f"Written: {outpath}")

        LOG.info("Running dispersion scan")
        for tds_voltage, parray_otrc64 in self.run_tds_scan():
            outpath = outdir / f"OTRC.64.I1D-voltage-{tds_voltage}.npz"
            save_particle_array(outpath, parray_otrc64)
            LOG.info(f"Written: {outpath}")

    def bolko_optics(self):
        mad8 = self.BOLKO_OPTICS
        df8 = pand8.read(mad8)

        controller = EuXFELController(components=_mad8_optics_to_magnet_config(df8))
        felconfig = FELSimulationConfig(controller=controller)

        # MAD8 uses name2s (i.e. distance from cathode wall not in the
        # name.  but ocelot lattice uses name1s.  so map here using the longlist
        start_name2 = df8.iloc[1].NAME
        stop_name2 = df8.iloc[-1].NAME

        start_name1 = self.longlist.name2_to_name1(start_name2).item()
        stop_name1 = self.longlist.name2_to_name1(stop_name2).item()

        twiss0 = Twiss()
        twiss0.beta_x = df8.iloc[0].BETX
        twiss0.beta_y = df8.iloc[0].BETY
        twiss0.alpha_x = df8.iloc[0].ALFX
        twiss0.alpha_y = df8.iloc[0].ALFY
        twiss0.s = self.b2dlat.get_element_start_s(start_name1)

        twiss, mlat = self.b2dlat.calculate_twiss(
            start=start_name1, stop=stop_name1, felconfig=felconfig, twiss0=twiss0
        )

        return twiss, mlat

    def _make_bolko_optics_config(self):
        mad8 = self.BOLKO_OPTICS
        df8 = pand8.read(mad8)

        felconfig = FELSimulationConfig(
            controller=EuXFELController(components=_mad8_optics_to_magnet_config(df8))
        )
        return felconfig

    def _full_dscan_configs(self, design_energy=False):
        bolko_config = self._make_bolko_optics_config()
        scan_quads = _dscan_conf_to_magnetic_config_dicts(self.dscan_conf)

        for scan_quads in _dscan_conf_to_magnetic_config_dicts(self.dscan_conf):
            nina_conf = deepcopy(bolko_config)
            if not design_energy:
                nina_conf.controller.a2.active = False
                nina_conf.controller.ah1.active = False
                nina_conf.controller.a3.active = False
                nina_conf.controller.a1.phi = 0
                nina_conf.controller.a1.v = 125 / 8 * 1e-3

            for quad_name, attrd in scan_quads.items():
                # Use bolko optics config as a baseline for nina's new config.
                nina_conf.controller.components[quad_name] = attrd

            yield nina_conf

    def new_optics_scan(self):
        for i, felconfig in enumerate(self._full_dscan_configs()):
            dy = self.dscan_conf.dispersions[i]

            mad8 = self.BOLKO_OPTICS
            df8 = pand8.read(mad8)

            # felconfig = FELSimulationConfig(components=_mad8_optics_to_magnet_config(df8, ll))

            # MAD8 uses name2s (i.e. distance from cathode wall not in the
            # name.  but ocelot lattice uses name1s.  so map here using the longlist
            start_name2 = df8.iloc[1].NAME
            stop_name2 = df8.iloc[-1].NAME

            start_name1 = self.longlist.name2_to_name1(start_name2)
            stop_name1 = self.longlist.name2_to_name1(stop_name2)

            twiss0 = Twiss()
            twiss0.beta_x = df8.iloc[0].BETX
            twiss0.beta_y = df8.iloc[0].BETY
            twiss0.alpha_x = df8.iloc[0].ALFX
            twiss0.alpha_y = df8.iloc[0].ALFY

            twiss, mlat = self.b2dlat.calculate_twiss(
                twiss0, start=start_name1, stop=stop_name1, felconfig=felconfig
            )

            yield dy, twiss

    def matching_point_to_screen(self, energy0):
        for i, felconfig in enumerate(self._full_dscan_configs()):
            dy = self.dscan_conf.dispersions[i]

            mad8 = self.BOLKO_OPTICS
            df8 = pand8.read(mad8)

            # felconfig = FELSimulationConfig(components=_mad8_optics_to_magnet_config(df8, ll))

            # MAD8 uses name2s (i.e. distance from cathode wall not in the
            # name.  but ocelot lattice uses name1s.  so map here using the longlist
            start_name2 = "BC2 matching point"
            stop_name2 = df8.iloc[-1].NAME

            stop_name1 = self.longlist.name2_to_name1(stop_name2)

            twiss0 = Twiss()
            twiss0.beta_x = df8.iloc[0].BETX
            twiss0.beta_y = df8.iloc[0].BETY
            twiss0.alpha_x = df8.iloc[0].ALFX
            twiss0.alpha_y = df8.iloc[0].ALFY
            twiss0.E = energy0
            twiss, mlat = self.b2dlat.calculate_twiss(
                twiss0, start=start_name2, stop=stop_name1, felconfig=felconfig
            )
            yield dy, twiss

    def optics_q52_to_b2_matching_point(self, twiss0):
        start_name = "matching-point-at-start-of-q52"
        stop_name = "BC2 matching point"
        full_twiss, mlat = self.b2dlat.calculate_twiss(
            twiss0,
            start=start_name,
            stop=stop_name,
            felconfig=next(self._full_dscan_configs()),
        )
        return full_twiss, mlat

    def design_optics(self):
        return self.b2dlat.calculate_twiss()

    def gun_to_dump_magnetic_lattice(self):
        return MagneticLattice(self.b2dlat.get_sequence())

    def gun_to_dump_sequence(self):
        return self.gun_to_dump_magnetic_lattice().sequence

    def gun_to_b2d_bolko_optics(self):
        mad8 = self.BOLKO_OPTICS
        df8 = pand8.read(mad8)

        felconfig = FELSimulationConfig(
            controller=EuXFELController(components=_mad8_optics_to_magnet_config(df8))
        )
        return self.b2dlat.calculate_twiss(felconfig=felconfig)

    def gun_to_dump_scan_optics(self, design_energy=False):
        for i, felconfig in enumerate(
            self._full_dscan_configs(design_energy=design_energy)
        ):
            dy = self.dscan_conf.dispersions[i]
            twiss, mlat = self.b2dlat.calculate_twiss(felconfig=felconfig)
            yield dy, twiss

    def gun_to_dump_piecewise_scan_optics(self):
        i = 0
        matching_points = [
            "matching-point-at-start-of-q52",
            "MATCH.174.L1",
            "MATCH.428.B2",
        ]  # "MATCH.414.B2",
        for felconfig, refconfig in zip(
            self._full_dscan_configs(design_energy=False),
            self._full_dscan_configs(design_energy=True),
        ):
            dy = self.dscan_conf.dispersions[i]
            felconfig.matching_config = refconfig
            felconfig.matching_points = matching_points
            twiss, mlat = self.b2dlat.calculate_twiss(
                felconfig=felconfig, design_config=refconfig
            )
            i += 1
            yield dy, twiss, matching_points

    def gun_to_dump_piecewise_scan_optics_tracking(
        self, design_energy=True, do_physics=False
    ):
        matching_points = [
            "matching-point-at-start-of-q52",
            "MATCH.174.L1",
            "MATCH.428.B2",
        ]
        i = 0
        for felconfig, refconfig in zip(
            self._full_dscan_configs(design_energy=False),
            self._full_dscan_configs(design_energy=True),
        ):
            dy = self.dscan_conf.dispersions[i]
            felconfig.matching_points = matching_points
            felconfig.do_physics = do_physics

            parray1, twiss = self.b2dlat.track_optics(
                self.parray0.copy(),
                start="G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity",
                felconfig=felconfig,
                design_config=refconfig,
            )
            i += 1
            yield dy, twiss, matching_points

    def gun_to_dump_central_slice_optics(
        self, design_energy=True, outdir=None, do_physics=False
    ):
        # do_
        matching_points = [
            "matching-point-at-start-of-q52",
            "MATCH.174.L1",
            "MATCH.428.B2",
        ]
        i = 0
        for felconfig, refconfig in zip(
            self._full_dscan_configs(design_energy=False),
            self._full_dscan_configs(design_energy=True),
        ):
            dy = self.dscan_conf.dispersions[i]
            felconfig.matching_points = matching_points  #
            felconfig.do_physics = do_physics
            dumps = [
                "OTRS.99.I1",
                "OTRS.192.B1" "OTRS.404.B2",
                "MATCH.428.B2",
                "MATCH.428.B2_after",
                "OTRA.473.B2D",
                "MATCH.37.I1",
                "MATCH.55.I1",
                "matching-point-at-start-of-q52",
            ]

            parray1, twiss = self.b2dlat.track_optics(
                self.parray0.copy(),
                dumps=dumps,
                start="G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity",
                felconfig=felconfig,
                opticscls=SliceTwissCalculator,
                design_config=refconfig,
                outdir=outdir,
            )
            i += 1
            yield dy, twiss, matching_points


class XFELLonglist:
    def __init__(self, path):
        self.ll = pd.read_csv(path)

    def name2_to_name1(self, name2):
        ldf = self.ll
        return ldf[ldf.NAME2 == name2].NAME1


def _mad8_optics_to_magnet_config(df8):
    quads = df8[df8.KEYWORD == "QUAD"]
    quad_name2s = quads.NAME
    quad_k1s = quads.K1

    longlist = XFELLonglist(LONGLIST)

    result = {}
    for name2, k1 in zip(quad_name2s, quad_k1s):
        name1s = longlist.name2_to_name1(name2)
        for name1 in name1s:
            result[name1] = {"k1": k1}

    sbends = df8[df8.KEYWORD == "SBEN"]
    for tup in sbends.itertuples():
        name2 = tup.NAME
        name1s = longlist.name2_to_name1(name2)
        for name1 in name1s:
            result[name1] = {"angle": tup.ANGLE, "e1": tup.E1, "e2": tup.E2}

    return result


def a1_to_i1d_design_optics():
    start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
    i1dlat = lattice.make_to_i1d_lattice(make_twiss0_at_a1_start())
    all_twiss, mlat = i1dlat.calculate_twiss(start=start_name)

    return all_twiss, mlat


def a1_to_q52_matching_point_measurement_optics():
    start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
    # start_name = "astra_ocelot_interface_A1_section_start"
    stop_name = "matching-point-at-start-of-q52"

    i1dlat = lattice.make_to_i1d_lattice(make_twiss0_at_a1_start())
    felconfig = make_pre_qi52_measurement_config()
    return i1dlat.calculate_twiss(start=start_name, stop=stop_name, felconfig=felconfig)


def qi52_matching_point_to_i1d_measurement_optics(dscan_quad_settings):
    i1dlat = lattice.make_to_i1d_lattice(make_twiss0_at_a1_start())
    start = "matching-point-at-start-of-q52"
    stop = None
    twiss0 = make_twiss_at_q52()

    fel_sim_configs = make_i1_dscan_simulation_configs(dscan_quad_settings)
    yield from _calculate_dscan_optics(
        i1dlat, twiss0, fel_sim_configs, start=start, stop=stop
    )


def qi52_matching_point_to_i1d_measurement_optics_tracking(
    parray0, dscan_quad_settings
):
    i1dlat = lattice.make_to_i1d_lattice()
    start = "matching-point-at-start-of-q52"
    stop = None

    fel_sim_configs = make_i1_dscan_simulation_configs(dscan_quad_settings)
    yield from _calculate_dscan_optics_tracking(
        i1dlat, parray0, fel_sim_configs, start=start, stop=stop
    )


def a1_dscan_piecewise_optics(dscan_conf, do_physics=False):
    twiss_to_q52, mlat_to_q52 = a1_to_q52_matching_point_measurement_optics()

    i1dlat = lattice.make_to_i1d_lattice(make_twiss_at_q52())
    start = "matching-point-at-start-of-q52"
    stop = None
    twiss_at_q52 = make_twiss_at_q52()

    fel_sim_configs = make_i1_dscan_simulation_configs(
        dscan_conf, do_physics=do_physics
    )
    for dispersion, twiss_from_q52, mlat_from_q52 in _calculate_dscan_optics(
        i1dlat, twiss_at_q52, fel_sim_configs, start=start, stop=stop
    ):
        piecewise_twiss = pd.concat([twiss_to_q52, twiss_from_q52])
        yield dispersion, piecewise_twiss


def a1_dscan_piecewise_tracked_optics(fparray0, dscan_conf, do_physics=False):
    (
        twiss_to_q52,
        parray_q52,
    ) = a1_to_q52_matching_point_measurement_optics_from_tracking(
        fparray0, do_physics=do_physics
    )

    i1dlat = lattice.make_to_i1d_lattice()
    start = "matching-point-at-start-of-q52"
    stop = None
    make_twiss_at_q52()

    fel_sim_configs = make_i1_dscan_simulation_configs(
        dscan_conf, do_physics=do_physics
    )
    for dispersion, twiss_from_q52, mlat_from_q52 in _calculate_dscan_tracked_optics(
        i1dlat, parray_q52.copy(), fel_sim_configs, start=start, stop=stop
    ):
        piecewise_twiss = pd.concat([twiss_to_q52, twiss_from_q52])
        yield dispersion, piecewise_twiss


def i1d_design_optics_from_tracking(fparray0):
    parray0 = load_particle_array(fparray0)

    # twiss, _ = cathode_to_first_a1_cavity_optics()
    # twiss0 = Twiss.from_series(twiss.iloc[-1])

    i1dlat = lattice.make_to_i1d_lattice(make_twiss0_at_a1_start())
    start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
    conf = FELSimulationConfig(do_physics=False)
    # all_twiss = i1dlat.track_gaussian_optics(twiss0, start=start_name, felconfig=conf, correct_dispersion=True)
    all_twiss = i1dlat.track_optics(
        parray0.copy(), opticscls=TwissCalculator, start=start_name, felconfig=conf
    )
    return all_twiss


def a1_to_q52_matching_point_measurement_optics_from_tracking(
    fparray0, do_physics=False
):
    parray0 = load_particle_array(fparray0)
    cathode_to_a1_twiss, _ = cathode_to_first_a1_cavity_optics()

    a1_twiss = cathode_to_a1_twiss.iloc[-1]

    # parray0_twiss = twiss_from_parray(parray0)

    i1dlat = lattice.make_to_i1d_lattice()
    start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
    stop_name = "matching-point-at-start-of-q52"
    conf = FELSimulationConfig(do_physics=do_physics)
    conf.ah1.active = False
    conf.a1.phi = 0
    conf.a1.v = 125 / 8 * 1e-3

    all_twiss, parray1 = i1dlat.track_gaussian_optics(
        a1_twiss, parray0=parray0, start=start_name, stop=stop_name, felconfig=conf
    )

    return all_twiss, parray1


# def qi52_matching_point_to_i1d_measurement_optics(fparray0):
#     parray0 = load_particle_array(fparray0)
#     cathode_to_a1_twiss, _ = cathode_to_first_a1_cavity_optics()
# # TO DO
#     a1_twiss = cathode_to_a1_twiss.iloc[-1]

#     # parray0_twiss = twiss_from_parray(parray0)

#     i1dlat = lattice.make_to_i1d_lattice()
#     start_name = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
#     stop_name = "matching-point-at-start-of-q52"
#     conf = FELSimulationConfig(do_physics=False)
#     conf.ah1.active = False
#     conf.a1.phi = 0
#     conf.a1.v = 125 / 8 * 1e-3

#     all_twiss = i1dlat.track_gaussian_optics(a1_twiss,
#                                              parray0=parray0,
#                                              start=start_name,
#                                              stop=stop_name,
#                                              felconfig=conf)

#     return all_twiss


# def parray_from_twiss0(twiss0):
#     energy = twiss0.E
#     from ocelot.common.globals import m_e_GeV
#     cov = cov_matrix_from_twiss(0.64e-6 / (energy / m_e_GeV),
#                                 0.64e-6 / (energy / m_e_GeV),
#                                 sigma_tau=0.0012803261741532739,
#                                 sigma_p=0.0054180378927533536,
#                                 beta_x=twiss0.beta_x.item(),
#                                 beta_y=twiss0.beta_y.item(),
#                                 alpha_x=twiss0.alpha_x.item(),
#                                 alpha_y=twiss0.alpha_y.item())

#     parray0 = cov_matrix_to_parray([0, 0, 0, 0, 0, 0], cov, energy, charge=250e-12, nparticles=200_000)
#     return parray0


def twiss_from_parray(parray):
    mean, cov = moments_from_parray(parray)
    parray_twiss = optics_from_moments(mean, cov, parray.E)
    return parray_twiss


# def calculate_dscan_optics_from_tracking(fparray0):
#     parray0 = load_particle_array(fparray0)

#     for setpoint_sim_conf in dscan_sim_configs:
#         twiss0 = make_twiss_at_q52()
#         dispersion = setpoint_sim_conf.metadata["dispersion"]
#         full_twiss, mlat = lat.calculate_twiss(twiss0, start=start, stop=stop,
#                                                felconfig=setpoint_sim_conf)

#         yield dispersion, full_twiss, lat
