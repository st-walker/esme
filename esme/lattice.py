"""Used for calibrating the measurement and also deriving measurement."""

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

import esme.i1 as i1
import esme.i1d as i1d

# from ocelot.cpbd.beam import Twiss
# from ocelot.cpbd.csr import CSR
# from ocelot.cpbd.magnetic_lattice import MagneticLattice
# from ocelot.cpbd.physics_proc import BeamTransform, LaserModulator, SaveBeam, SmoothBeam
# from ocelot.cpbd.sc import SpaceCharge
# from ocelot.cpbd.wake3D import Wake, WakeTable
# from ocelot.utils.section_track import SectionTrack


LOG = logging.getLogger(__name__)

# import esme.l1 as l1

# # Sig_Z=(0.0019996320155001497, 0.0006893836215002082, 0.0001020391309281775, 1.25044082708419e-05) #500pC 5kA
# # Sig_Z=(0.0019996320155001497, 0.0006817907866411071, 9.947650872824487e-05, 7.13045869665955e-06)  #500pC 10kA
# # Sig_Z=(0.0018761888067590127, 0.0006359220169656093, 9.204477386791353e-05, 7.032551498646372e-06) #250pC 5kA
# # Sig_Z=(0.0018856911379360524, 0.0005463919476045524, 6.826162032352288e-05, 1.0806534547678727e-05) #100pC 1kA
# Sig_Z = (0.0018732376720197858, 0.000545866016784069, 7.09234589639138e-05, 2.440742745010469e-06)  # 100 pC 5kA
# # Sig_Z=(0.0013314283765668853, 0.0004502566926198658, 4.64037216210807e-05, 2.346018397815618e-06) #100 pC 5kA SC
# # Sig_Z=(0.0013314187263949542, 0.00045069372029991764, 4.537451914820527e-05, 4.0554988027793585e-06)#100 pC 2.5kA SC


# SmoothPar = 1000
# LHE = 0 * 5000e-9  # GeV
# WakeSampling = 1000
# WakeFilterOrder = 10
# CSRBin = 400
# CSRSigmaFactor = 0.1
# SCmesh = [63, 63, 63]
# bISR = True
# bRandomMesh = True


# class A1(SectionTrack):
#     def __init__(self, data_dir, *args, **kwargs):
#         super().__init__(data_dir)
#         # setting parameters
#         self.lattice_name = 'A1'
#         self.unit_step = 0.02
#         self.input_beam_file = self.particle_dir + 'Exfel.0320.ast'
#         self.output_beam_file = self.particle_dir + 'section_A1.npz'
#         self.tws_file = self.tws_dir + "tws_section_A1.npz"
#         # init tracking lattice
#         start_sim = i1.start_sim
#         acc1_stop = i1.a1_sim_stop
#         self.lattice = MagneticLattice(i1.cell, start=start_sim, stop=acc1_stop, method=self.method)
#         # init physics processes
#         sc = SpaceCharge()
#         sc.step = 1
#         sc.nmesh_xyz = SCmesh
#         sc.random_mesh = bRandomMesh
#         sc2 = SpaceCharge()
#         sc2.step = 1
#         sc2.nmesh_xyz = SCmesh
#         sc2.random_mesh = bRandomMesh
#         wake = Wake()
#         wake.wake_table = WakeTable('accelerator/wakes/RF/wake_table_A1.dat')
#         wake.factor = 1
#         wake.step = 10
#         wake.w_sampling = WakeSampling
#         wake.filter_order = WakeFilterOrder
#         smooth = SmoothBeam()
#         smooth.mslice = SmoothPar
#         # adding physics processes
#         acc1_1_stop = i1.a1_1_stop
#         self.add_physics_process(smooth, start=start_sim, stop=start_sim)
#         self.add_physics_process(sc, start=start_sim, stop=acc1_1_stop)
#         self.add_physics_process(sc2, start=acc1_1_stop, stop=acc1_stop)
#         self.add_physics_process(wake, start=i1.c_a1_1_1_i1, stop=acc1_stop)


# class AH1(SectionTrack):
#     def __init__(self, data_dir, *args, **kwargs):
#         super().__init__(data_dir)
#         # setting parameters
#         self.lattice_name = 'Injector AH1'
#         self.unit_step = 0.02
#         self.input_beam_file = self.particle_dir + 'section_A1.npz'
#         self.output_beam_file = self.particle_dir + 'section_AH1.npz'
#         self.tws_file = self.tws_dir + "tws_section_AH1.npz"
#         # init tracking lattice
#         acc1_stop = i1.a1_sim_stop
#         acc39_stop = i1.stlat_47_i1
#         self.lattice = MagneticLattice(i1.cell, start=acc1_stop, stop=acc39_stop, method=self.method)
#         # init physics processes
#         sc = SpaceCharge()
#         sc.step = 5
#         sc.nmesh_xyz = SCmesh
#         sc.random_mesh = bRandomMesh
#         wake = Wake()
#         wake.wake_table = WakeTable('accelerator/wakes/RF/wake_table_AH1.dat')
#         wake.factor = 1
#         wake.step = 10
#         wake.w_sampling = WakeSampling
#         wake.filter_order = WakeFilterOrder
#         # adding physics processes
#         self.add_physics_process(sc, start=acc1_stop, stop=acc39_stop)
#         self.add_physics_process(wake, start=i1.c3_ah1_1_1_i1, stop=acc39_stop)


# class LH(SectionTrack):
#     def __init__(self, data_dir, *args, **kwargs):
#         super().__init__(data_dir)
#         # setting parameters
#         self.lattice_name = 'LASER HEATER MAGNETS'
#         self.unit_step = 0.02
#         self.input_beam_file = self.particle_dir + 'section_AH1.npz'
#         self.output_beam_file = self.particle_dir + 'section_LH.npz'
#         self.tws_file = self.tws_dir + "tws_section_LH.npz"
#         # init tracking lattice
#         acc39_stop = i1.stlat_47_i1
#         # lhm_stop = l1.id_90904668_ #for s2e
#         # lhm_stop = l1.cix_65_i1    #approx. corresponds to the position of the screen in I1D.
#         lhm_stop = i1.dump_csr_start  # for going in I1D
#         self.lattice = MagneticLattice(i1.cell + l1.cell, start=acc39_stop, stop=lhm_stop, method=self.method)
#         # init physics processes
#         csr = CSR()
#         csr.sigma_min = Sig_Z[0] * CSRSigmaFactor
#         csr.traj_step = 0.0005
#         csr.apply_step = 0.005
#         sc = SpaceCharge()
#         sc.step = 5
#         sc.nmesh_xyz = SCmesh
#         sc.random_mesh = bRandomMesh
#         wake = Wake()
#         wake.wake_table = WakeTable('accelerator/wakes/RF/wake_table_TDS1.dat')
#         wake.factor = 1
#         wake.step = 10
#         wake.w_sampling = WakeSampling
#         wake.filter_order = WakeFilterOrder

#         lh = LaserModulator()
#         lh.dE = LHE
#         lh.sigma_l = 300
#         lh.sigma_x = 300e-6
#         lh.sigma_y = 300e-6
#         lh.z_waist = None

#         filename_tds1 = self.particle_dir + "tds1.npz"
#         filename_tds2 = self.particle_dir + "tds2.npz"
#         filename_tds3 = self.particle_dir + "tds3.npz"
#         if "suffix" in kwargs:
#             filename, file_extension = os.path.splitext(filename_tds1)
#             filename_tds1 = filename + str(kwargs["suffix"]) + file_extension
#             filename, file_extension = os.path.splitext(filename_tds2)
#             filename_tds2 = filename + str(kwargs["suffix"]) + file_extension
#             filename, file_extension = os.path.splitext(filename_tds3)
#             filename_tds3 = filename + str(kwargs["suffix"]) + file_extension

#         sv_tds1 = SaveBeam(filename=filename_tds1)
#         # sv_tds2 = SaveBeam(filename=filename_tds2)
#         # sv_tds3 = SaveBeam(filename=filename_tds3)

#         tws_52 = Twiss()
#         tws_52.beta_x = 3.131695851
#         tws_52.beta_y = 5.417462794
#         tws_52.alpha_x = -0.9249364470
#         tws_52.alpha_y = 1.730107901
#         tws_52.gamma_x = (1 + tws_52.alpha_x**2) / tws_52.beta_x

#         tr = BeamTransform(tws=tws_52)
#         # tr.bounds = [-0.5, 0.5]
#         tr.slice = "Emax"
#         self.add_physics_process(sc, start=acc39_stop, stop=lhm_stop)
#         self.add_physics_process(csr, start=acc39_stop, stop=lhm_stop)
#         self.add_physics_process(wake, start=acc39_stop, stop=lhm_stop)
#         self.add_physics_process(sv_tds1, start=i1.tds1, stop=i1.tds1)
#         # self.add_physics_process(sv_tds2, start=i1.tds2, stop=i1.tds2)
#         # self.add_physics_process(sv_tds3, start=i1.tds3, stop=i1.tds3)
#         self.add_physics_process(tr, i1.tmp_m, i1.tmp_m)


# class I1D_Screen(SectionTrack):
#     def __init__(self, data_dir, *args, **kwargs):
#         super().__init__(data_dir)
#         # setting parameters
#         self.lattice_name = 'I1 DUMP'
#         self.unit_step = 0.02
#         self.input_beam_file = self.particle_dir + 'section_LH.npz'
#         self.output_beam_file = self.particle_dir + 'section_I1D.npz'
#         self.tws_file = self.tws_dir + "tws_section_I1D.npz"
#         filename_dump = self.particle_dir + "dump.npz"
#         if "suffix" in kwargs:
#             filename, file_extension = os.path.splitext(self.output_beam_file)
#             self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
#             filename, file_extension = os.path.splitext(self.tws_file)
#             self.tws_file = filename + str(kwargs["suffix"]) + file_extension
#             filename, file_extension = os.path.splitext(filename_dump)
#             filename_dump = filename + str(kwargs["suffix"]) + file_extension

#         sv_dump = SaveBeam(filename=filename_dump)
#         # init tracking lattice
#         i1d_start = i1.dump_csr_start
#         i1d_stop = i1d.otrc_64_i1d
#         # i1d_stop = i1d.stsec_62_i1d
#         self.lattice = MagneticLattice(i1.cell + i1d.cell, start=i1d_start, stop=i1d_stop, method=self.method)
#         # init physics processes
#         sigma = Sig_Z[0]
#         csr = CSR()
#         csr.sigma_min = sigma * 0.1
#         csr.traj_step = 0.0005
#         csr.apply_step = 0.005

#         sc = SpaceCharge()
#         sc.step = 5
#         sc.nmesh_xyz = SCmesh
#         sc.random_mesh = bRandomMesh

#         self.add_physics_process(sc, start=i1d_start, stop=i1d_stop)
#         self.add_physics_process(csr, start=i1d_start, stop=i1d.bpma_63_i1d)
#         self.add_physics_process(sv_dump, start=i1d_start, stop=i1d_start)


def design_quad_strengths() -> pd.DataFrame:
    # This is all very hardcoded for the measurement we did in October 2022.
    # Get data in correct shape for passing to pd.DataFrame
    # I think these are what they're "supposed" to be, should really check...
    dispersions = [1.197, 1.025, 0.801, 0.6590]
    # These are quads that are set at the start of the measurement and then left
    # constant for the whole scan.

    upstream_quad_names = [
        "QI.52.I1",
        "QI.53.I1",
        "QI.54.I1",
        "QI.55.I1",
        "QI.57.I1",
        "QI.59.I1",
        # "QI.60.I1", "QI.61.I1", "QI.63.I1D", "QI.64.I1D"
    ]
    # These are K1Ls but in mrad/m. whereas k1l is "usually" rad/m. These are
    # the upstream/matching quads and stay the same for the whole scan.
    upstream_strengths = [
        -83.7203,
        500.308,
        188.82,
        -712.48,
        712.48,
        -712.48,
        # Commented out as they are set below in scan_strengths.
        # -509.559, 832.63, -249.585, 831.95
    ]

    # These are the quads which change once per point in the dispersion scan.
    scan_quad_names = ["QI.60.I1", "QI.61.I1", "QI.63.I1D", "QI.64.I1D"]
    scan_strengths = [
        [-509.559, 832.63, -249.585, 831.95],
        [-509.0345, 832.175, 106.965, 475.4],
        [-508.965, 820.2076, 582.365, 0],
        [-508.749, 789.625, 1046.306, -475.4],
    ]

    result = []

    for dx, strengths in zip(dispersions, scan_strengths):
        strengths = upstream_strengths + strengths
        stds = np.zeros_like(strengths)
        quad_names = upstream_quad_names + scan_quad_names
        df = pd.DataFrame(np.vstack((strengths, stds)).T, index=quad_names, columns=["kick_mean", "kick_std"])

        df = _append_element_positions_and_names(df, quad_names)
        result.append(df)

    return dispersions, result


def _append_element_positions_and_names(df: pd.DataFrame, names: Iterable[str]):
    cell = injector_cell()
    lengths = []
    positions = []
    for name in names:
        try:
            length = cell.element_attributes(name, "l")
        except KeyError:
            length = np.nan
        try:
            s = cell.element_s(name)
        except KeyError:
            s = np.nan

        lengths.append(length)
        positions.append(s)
    return df.assign(length=lengths, s=positions)


def mean_quad_strengths(df: pd.DataFrame, include_s=True, dropna=True):
    pattern = re.compile(r"Q(I?)|(L[NS])\.")
    quad_names = [key for key in df.keys() if pattern.match(key)]

    df_quads = df[quad_names]

    df = pd.DataFrame({"kick_mean": df_quads.mean(), "kick_std": df_quads.std()})

    df = _append_element_positions_and_names(df, quad_names)

    if dropna:
        return df.dropna()
    return df


def injector_cell():
    return i1.make_cell() + i1d.make_cell()


def injector_cell_from_snapshot(snapshot: pd.Series, check=True, change_correctors=False):
    quad_mask = snapshot.index.str.contains(r"^Q(?:I?|L[NS])\.")
    solenoid_mask = snapshot.index.str.contains(r"^SOL[AB]\.")
    corrector_mask = snapshot.index.str.contains(r"^C[IK]?[XY]\.")
    bends_mask = snapshot.index.str.contains(r"^B[BL]\.")
    used_mask = quad_mask | solenoid_mask | corrector_mask | corrector_mask | bends_mask

    if check:
        xfel_mask = snapshot.index.str.startswith("XFEL.")
        bpm_mask = snapshot.index.str.contains(r"^BPM[GAFRSCD]\.")
        timestamp_mask = snapshot.index == "timestamp"
        sextupole_mask = snapshot.index.str.contains(r"^SC\.")
        other_kickers_mask = snapshot.index.str.contains(r"CB[LB]\.")
        unused_mask = xfel_mask | bpm_mask | timestamp_mask | sextupole_mask | other_kickers_mask
        my_added_metadata = snapshot.index.str.startswith("MY_")
        the_rest_mask = ~(used_mask | unused_mask | my_added_metadata)
        snapshot[the_rest_mask]

        the_rest = snapshot[the_rest_mask]

        expected = {"RF.23.I1", "BK.24.I1"}
        if set(the_rest.index) != expected:
            LOG.warning("Unexpected item in DF.  Expected: {expected}, got: {the_rest}")

    cell = injector_cell()
    for quad_name, int_strength in snapshot[quad_mask].items():
        try:
            quad = cell[quad_name]
        except KeyError:
            LOG.debug("NOT setting quad strength from snapshot: %s", quad_name)
        else:
            k1 = int_strength * 1e-3 / quad.l  # Convert to rad/m from mrad/m then to k1.
            LOG.debug(f"setting quad strength from snapshot: {quad_name=}, {quad.k1=} to {k1=}")
            quad.k1 = k1

    dipole_mask = bends_mask
    if change_correctors:
        dipole_mask |= corrector_mask
    for dipole_name, angle_mrad in snapshot[dipole_mask].items():
        try:
            dipole = cell[dipole_name]
        except KeyError:
            LOG.debug("NOT setting dipole angle from snapshot: %s", dipole_name)
        else:
            # I insert a negative sign here.  I guess convention is opposite?
            angle = -angle_mrad * 1e-3
            LOG.debug(f"setting dipole angle from snapshot: {dipole_name=}, {dipole.angle=} to {angle=}")
            dipole.angle = angle

    return cell
