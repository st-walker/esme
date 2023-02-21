from pathlib import Path

from . import i1
from . import i1d
from . import l1
from . import l2
from . import b2d

from ocelot.cpbd.beam import Twiss
from ocelot.cpbd.csr import CSR
from ocelot.cpbd.magnetic_lattice import MagneticLattice
from ocelot.cpbd.physics_proc import BeamTransform, LaserModulator, SaveBeam, SmoothBeam
from ocelot.cpbd.sc import SpaceCharge
from ocelot.cpbd.wake3D import Wake, WakeTable
from ocelot.utils.fel_track import FELSection, SectionedFEL


BEAM_SMOOTHER = SmoothBeam()
BEAM_SMOOTHER.mslice = 1000

LHE = 11000e-9 * 0.74 / 0.8  # GeV
WAKE_SAMPLING = 1000
WAKE_FILTER_ORDER = 10
CSR_SIGMA_FACTOR = 0.1
SC_MESH = [63, 63, 63]
# bISR = True
SC_RANDOM_MESH = True
CSR_N_BIN = 400


# Sig_Z=(0.0019996320155001497, 0.0006893836215002082, 0.0001020391309281775, 1.25044082708419e-05) #500pC 5kA
# Sig_Z=(0.0019996320155001497, 0.0006817907866411071, 9.947650872824487e-05, 7.13045869665955e-06)  #500pC 10kA
# Sig_Z=(0.0018761888067590127, 0.0006359220169656093, 9.204477386791353e-05, 7.032551498646372e-06) #250pC 5kA
# Sig_Z=(0.0018856911379360524, 0.0005463919476045524, 6.826162032352288e-05, 1.0806534547678727e-05) #100pC 1kA
Sig_Z = (0.0018732376720197858, 0.000545866016784069, 7.09234589639138e-05, 2.440742745010469e-06)  # 100 pC 5kA
# Sig_Z=(0.0013314283765668853, 0.0004502566926198658, 4.64037216210807e-05, 2.346018397815618e-06) #100 pC 5kA SC
# Sig_Z=(0.0013314187263949542, 0.00045069372029991764, 4.537451914820527e-05, 4.0554988027793585e-06)#100 pC 2.5kA SC


def make_space_charge(*, step, nmesh_xyz=None, random_mesh=None):
    sc = SpaceCharge()
    sc.step = step
    if nmesh_xyz is not None:
        sc.nmesh_xyz = nmesh_xyz
    if random_mesh is not None:
        sc.random_mesh = random_mesh
    return sc


def make_wake(rel_path, *, factor, step=None, w_sampling=None, filter_order=None):
    wake = Wake()
    wake.wake_table = WakeTable(Path(__file__).parent / rel_path)
    wake.factor = factor
    if step is not None:
        wake.step = step

    if w_sampling is not None:
        wake.w_sampling = w_sampling
    if filter_order is not None:
        wake.filter_order = filter_order

    return wake


def make_beam_transform(*, betx, bety, alfx, alfy, tr_slice):
    tws = Twiss()
    tws.beta_x = betx
    tws.beta_y = bety
    tws.alpha_x = alfx
    tws.alpha_y = alfy
    tws.gamma_x = (1 + alfx**2) / betx
    tr = BeamTransform(tws=tws)
    tr.slice = tr_slice
    return tr


def make_laser_modulator():
    lh = LaserModulator()
    lh.dE = 300
    lh.sigma_l = 300
    lh.sigma_x = 300e-6
    lh.sigma_y = 300e-6
    lh.z_waist = None
    return lh


def make_csr(*, sigma_min, traj_step, apply_step, n_bin=None, step=None):
    csr = CSR()
    csr.sigma_min = sigma_min
    csr.traj_step = traj_step
    csr.apply_step = apply_step
    if n_bin is not None:
        csr.n_bin = n_bin
    if step is not None:
        csr.step = step
    return csr



class G1(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # This section is here for completion and clarity.  It should
        # not be used for tracking.  At most it should be used for optics.
        # Force a bomb here so no tracking can take place with this FELSection
        self.unit_step = "Hello please don't use G1 for tracking"

        i1_cell = i1.make_cell()
        # Start at the beginning of I1.
        # gun_start = None
        # This is also simply directly in front of the first A1 cavity.
        # gun_stop = "astra_ocelot_interface_G1_to_A1"
        gun_stop = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
        self.sequence = i1_cell[:gun_stop]
        # self.lattice = MagneticLattice(i1_cell, start=gun_start, stop=gun_stop)


class A1(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.unit_step = 0.02
        # init tracking lattice
        i1_cell = i1.make_cell()

        # Simply the start, just after cathode.  3.2m into the
        # sequence.  This is where the parray is tracked to in astra.
        # a1_start = i1_cell["OCELOT_ASTRA_INTERFACE"]
        # a1_start = i1_cell["ID_22433449_"]
        # a1_start = None
        a1_start = "G1-A1 interface: up to where we track using ASTRA and just right the first A1 cavity"
        # a1_start = "astra_ocelot_interface_G1_to_A1"
        # just after last cavity module of A1
        a1_stop = "A1-AH1 interface: just after the last A1 Cavity"
        # a1_stop = i1_cell["ID_68749308_"]
        self.sequence = i1_cell[a1_start:a1_stop]

        # init physics processes
        sc = make_space_charge(step=1, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)
        sc2 = make_space_charge(step=1, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)
        wake = make_wake(
            'wake_table_A1.dat', factor=1, step=10, w_sampling=WAKE_SAMPLING, filter_order=WAKE_FILTER_ORDER
        )
        # adding physics processes
        # just after the first A1 cavity.
        just_after_first_a1_cavity = i1_cell["just-after-first-a1-cavity"]
        first_cavity = i1_cell["C.A1.1.1.I1"]

        # beam is immediately smoothed right at the start of the simulation in an instant
        self.add_physics_process(BEAM_SMOOTHER, start=a1_start, stop=a1_start)

        # Attach a SC instance from start (just after gun) to just after first module
        self.add_physics_process(sc, start=a1_start, stop=just_after_first_a1_cavity)
        # Attach a different SC instance between before 2nd module and end of last module.
        self.add_physics_process(sc2, start=just_after_first_a1_cavity, stop=a1_stop)
        # From start of A1 cavities to end of A1 (just after last cavity), attach wake kick.
        self.add_physics_process(wake, start=first_cavity, stop=a1_stop)


class AH1(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.unit_step = 0.02
        # init tracking lattice
        i1_cell = i1.make_cell()
        # just after last cavity module of A1 (see class A1 above)
        ah1_section_start = "A1-AH1 interface: just after the last A1 Cavity"

        # This is just before the first laser heater chicane magnet.
        # It is a bit after the last high order cavity (i.e. of AH1)
        # Not directly before because we want physics to be correctly in the next section.
        # just_before_lh_first_dipole = i1_cell["just-before-first-laser-heater-dipole"]
        ah1_section_stop = "AH1-LH interface: Just before the first LH chicane dipole"
        # just_before_lh_first_dipole =
        self.sequence = i1_cell[ah1_section_start:ah1_section_stop]
        # init physics processes
        sc = make_space_charge(step=5, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)
        wake = make_wake(
            "wake_table_AH1.dat", factor=1, step=10, w_sampling=WAKE_SAMPLING, filter_order=WAKE_FILTER_ORDER
        )

        # adding physics processes
        self.add_physics_process(sc, start=ah1_section_start, stop=ah1_section_stop)
        first_high_order_cavity = i1_cell["C3.AH1.1.1.I1"]
        self.add_physics_process(wake, start=first_high_order_cavity, stop=ah1_section_stop)


class LH(FELSection):
    """Importantly this section includes the TDS!"""

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.unit_step = 0.02
        # init tracking lattice

        i1_cell = i1.make_cell()
        # Start where AH1 ends (see AH1 class)
        lh_section_start = "AH1-LH interface: Just before the first LH chicane dipole"

        # pre_tds_matching_point = i1_cell["pre_qi_52"]
        just_before_lh_first_dipole = i1_cell["STLAT.47.I1"]
        # lh_section_start = i1_cell["just_before_first_laser_heater_dipole_LH_section_start"]
        # lh_section_stop = i1_cell["just_before_i1d_dipole_LH_section_stop"]

        lh_section_stop = "LH-I1D interface: just before the I1D dump dipole"

        # Where we start the CSR process if we are going into the
        # dump. This is just before the dump dipole BB.62.I1D.
        just_before_dump_dipole = "DUMP.CSR.START"
        self.sequence = i1_cell[lh_section_start:lh_section_stop]

        # init physics processes
        csr = make_csr(sigma_min=Sig_Z[0] * CSR_SIGMA_FACTOR, traj_step=0.0005, apply_step=0.005)
        sc = make_space_charge(step=5, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)
        wake = make_wake(
            "wake_table_TDS1.dat", factor=1, step=10, w_sampling=WAKE_SAMPLING, filter_order=WAKE_FILTER_ORDER
        )
        lh = make_laser_modulator()
        tr = make_beam_transform(
            betx=3.131695851, bety=5.417462794, alfx=-0.9249364470, alfy=1.730107901, tr_slice="Emax"
        )

        self.add_physics_process(sc, start=just_before_lh_first_dipole, stop=just_before_dump_dipole)
        self.add_physics_process(csr, start=just_before_lh_first_dipole, stop=just_before_dump_dipole)
        self.add_physics_process(wake, start=just_before_lh_first_dipole, stop=just_before_dump_dipole)
        # pre_tds_matching_point = i1_cell["matching-point-at-start-of-q52"]
        # self.add_physics_process(tr, pre_tds_matching_point, pre_tds_matching_point)


class I1D(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.unit_step = 0.02

        filename_dump = self.outdir_particles / "OTRC.64.I1D.npz"

        otrc_save = SaveBeam(filename=filename_dump)

        i1_cell = i1.make_cell()
        i1d_cell = i1d.make_cell()
        cell = i1_cell + i1d_cell

        # i1d_start = "just_before_i1d_dipole_I1D_section_start"        
        i1d_start = "LH-I1D interface: just before the I1D dump dipole"
        self.sequence = cell[i1d_start:]

        otrc_marker = cell["OTRC.64.I1D"]
        post_dump_dipole_bpm = cell["BPMA.63.I1D"]
        # Just track all the way to the end of the dump line
        stop = None
        # self.lattice = MagneticLattice(cell, start=i1d_start, stop=stop, method=self.method)
        # init physics processes
        csr = make_csr(sigma_min=Sig_Z[0] * 0.1, traj_step=0.0005, apply_step=0.005)
        sc = make_space_charge(step=5, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)

        # Add physics processes.  SC the whole way, CSR only for the dipole (more or less).
        self.add_physics_process(sc, start=i1d_start, stop=stop)
        self.add_physics_process(csr, start=i1d_start, stop=post_dump_dipole_bpm)
        # Write particle array to file at the screen.
        self.add_physics_process(otrc_save, start=otrc_marker, stop=otrc_marker)


class DL(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        self.unit_step = 0.02
        # st2_stop = l1.id_90904668_

        i1_cell = i1.make_cell()
        l1_cell = l1.make_cell()
        cell = i1_cell + l1_cell

        # End of LH (so start of DL...).  This is just before the dump dipole to I1D.
        lh_stop_dl_start = cell["DUMP.CSR.START"]
        # Just before the first BC0 dipole:
        dogleg_stop = cell["STLAT.96.I1"]
        self.lattice = MagneticLattice(cell, start=lh_stop_dl_start, stop=dogleg_stop, method=self.method)
        # init physics processes
        csr = make_csr(sigma_min=Sig_Z[0] * CSR_SIGMA_FACTOR, traj_step=0.0005, apply_step=0.005, n_bin=CSR_N_BIN)
        wake = make_wake(
            'mod_wake_0070.030_0073.450_MONO.dat',
            factor=1,
            w_sampling=WAKE_SAMPLING,
            filter_order=WAKE_FILTER_ORDER,
        )
        sc = make_space_charge(step=25, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)

        self.add_physics_process(csr, start=lh_stop_dl_start, stop=dogleg_stop)
        self.add_physics_process(sc, start=lh_stop_dl_start, stop=dogleg_stop)
        self.add_physics_process(wake, start=dogleg_stop, stop=dogleg_stop)


class BC0(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'BC0'
        self.unit_step = 0.02

        l1_cell = l1.make_cell()

        dogleg_stop_bc0_start = l1_cell["STLAT.96.I1"]
        bc0_stop = l1_cell["ENLAT.101.I1"]
        self.lattice = MagneticLattice(l1_cell, start=dogleg_stop_bc0_start, stop=bc0_stop, method=self.method)

        csr = make_csr(
            step=1, n_bin=CSR_N_BIN, sigma_min=Sig_Z[1] * CSR_SIGMA_FACTOR, traj_step=0.0005, apply_step=0.001
        )
        sc = make_space_charge(step=40, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)

        self.add_physics_process(sc, start=dogleg_stop_bc0_start, stop=bc0_stop)
        self.add_physics_process(csr, start=dogleg_stop_bc0_start, stop=bc0_stop)

        # self.dipoles = [l1.bb_96_i1, l1.bb_98_i1, l1.bb_100_i1, l1.bb_101_i1]
        # self.dipole_len = 0.5
        # self.bc_gap=1.0


class L1(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        self.unit_step = 0.02

        l1_cell = l1.make_cell()
        # Just after end of BC0
        bc0_stop = l1_cell["ENLAT.101.I1"]
        # This is just before the start of BC1.
        a2_stop_bc1_start = l1_cell["STLAT.182.B1"]

        self.lattice = MagneticLattice(l1_cell, start=bc0_stop, stop=a2_stop_bc1_start, method=self.method)

        sc = make_space_charge(step=50, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)
        wake = make_wake(
            "mod_TESLA_MODULE_WAKE_TAYLOR.dat",
            factor=4,
            step=100,
            w_sampling=WAKE_SAMPLING,
            filter_order=WAKE_FILTER_ORDER,
        )
        wake2 = make_wake(
            "mod_wake_0078.970_0159.280_MONO.dat", factor=1, w_sampling=WAKE_SAMPLING, filter_order=WAKE_FILTER_ORDER
        )

        l1_first_cavity = l1_cell["C.A2.1.1.L1"]
        l1_last_cavity = l1_cell["C.A2.4.8.L1"]
        self.add_physics_process(BEAM_SMOOTHER, start=bc0_stop, stop=bc0_stop)
        self.add_physics_process(sc, start=bc0_stop, stop=a2_stop_bc1_start)
        self.add_physics_process(wake, start=l1_first_cavity, stop=l1_last_cavity)
        self.add_physics_process(wake2, start=a2_stop_bc1_start, stop=a2_stop_bc1_start)


class BC1(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        self.unit_step = 0.02

        l1_cell = l1.make_cell()
        a2_stop_bc1_start = l1_cell["STLAT.182.B1"]
        bc1_stop = l1_cell["TORA.203.B1"]
        # init tracking lattice
        self.lattice = MagneticLattice(l1_cell, start=a2_stop_bc1_start, stop=bc1_stop, method=self.method)

        # init physics processes
        csr = make_csr(
            step=1, n_bin=CSR_N_BIN, sigma_min=Sig_Z[2] * CSR_SIGMA_FACTOR, traj_step=0.0005, apply_step=0.001
        )
        sc = make_space_charge(step=40, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)

        self.add_physics_process(csr, start=a2_stop_bc1_start, stop=bc1_stop)
        self.add_physics_process(sc, start=a2_stop_bc1_start, stop=bc1_stop)

        # self.dipoles = [l1.bb_182_b1, l1.bb_191_b1, l1.bb_193_b1, l1.bb_202_b1]
        # self.dipole_len = 0.5
        # self.bc_gap=8.5


class L2(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        self.unit_step = 0.02

        l1_cell = l1.make_cell()
        l2_cell = l2.make_cell()

        cell = l1_cell + l2_cell

        bc1_stop = cell["TORA.203.B1"]
        # Just before BC2
        l2_stop_bc2_start = cell["STLAT.393.B2"]

        # init tracking lattice
        self.lattice = MagneticLattice(cell, start=bc1_stop, stop=l2_stop_bc2_start, method=self.method)

        sc = make_space_charge(step=100, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)
        wake = make_wake("mod_TESLA_MODULE_WAKE_TAYLOR.dat", factor=4 * 3, step=200)
        wake2 = make_wake("mod_wake_0179.810_0370.840_MONO.dat", factor=1)

        first_cavity_l2 = cell["C.A3.1.1.L2"]
        last_cavity_l2 = cell["C.A5.4.8.L2"]

        self.add_physics_process(BEAM_SMOOTHER, start=bc1_stop, stop=bc1_stop)
        self.add_physics_process(sc, start=bc1_stop, stop=l2_stop_bc2_start)
        self.add_physics_process(wake, start=first_cavity_l2, stop=last_cavity_l2)
        self.add_physics_process(wake2, start=l2_stop_bc2_start, stop=l2_stop_bc2_start)


class BC2(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)

        self.unit_step = 0.02

        l2_cell = l2.make_cell()

        l2_stop_bc2_start = l2_cell["STLAT.393.B2"]
        bc2_stop = l2_cell["TORA.415.B2"]
        # init tracking lattice
        self.lattice = MagneticLattice(l2_cell, start=l2_stop_bc2_start, stop=bc2_stop, method=self.method)

        csr = make_csr(
            step=1, n_bin=CSR_N_BIN, sigma_min=Sig_Z[3] * CSR_SIGMA_FACTOR, traj_step=0.0005, apply_step=0.001
        )
        sc = make_space_charge(step=50, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)

        self.add_physics_process(csr, start=l2_stop_bc2_start, stop=bc2_stop)
        self.add_physics_process(sc, start=l2_stop_bc2_start, stop=bc2_stop)

        # self.dipoles = [l2.bb_393_b2, l2.bb_402_b2, l2.bb_404_b2, l2.bb_413_b2]
        # self.dipole_len = 0.5
        # self.bc_gap = 8.5


class B2D(FELSection):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        self.unit_step = 0.02

        l2_cell = l2.make_cell()
        b2d_cell = b2d.make_cell()

        cell = l2_cell + b2d_cell

        bc2_stop = cell["TORA.415.B2"]
        csr_start = bc2_stop
        b2d_stop = None

        self.lattice = MagneticLattice(cell, start=bc2_stop, stop=b2d_stop, method=self.method)

        csr = make_csr(
            step=1, n_bin=CSR_N_BIN, sigma_min=Sig_Z[3] * CSR_SIGMA_FACTOR, traj_step=0.0005, apply_step=0.001
        )
        sc = make_space_charge(step=50, nmesh_xyz=SC_MESH, random_mesh=SC_RANDOM_MESH)

        self.add_physics_process(csr, start=bc2_stop, stop=b2d_stop)
        self.add_physics_process(sc, start=bc2_stop, stop=b2d_stop)
