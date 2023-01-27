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
from ocelot.utils.section_track import SectionTrack, SectionLattice


BEAM_SMOOTHER = SmoothBeam()
BEAM_SMOOTHER.mslice = 1000

LHE = 11000e-9 * 0.74 / 0.8  # GeV
WakeSampling = 500
WakeFilterOrder = 20
CSRBin = 400
CSRSigmaFactor = 0.1
SCmesh = [63, 63, 63]
bISR = True
bRandomMesh = True



def make_space_charge(step, nmesh_xyz=None, random_mesh=None):
    sc = SpaceCharge()
    sc.step = step
    if nmesh_xyz is None:
        nmesh_xyz = SCmesh
    sc.nmesh_xyz = nmesh_xyz
    if random_mesh is not None:
        sc.random_mesh = random_mesh
    return sc

def make_wake(rel_path, factor, step, *, w_sampling=None, filter_order=None):
    wake = Wake()
    wake.wake_table = WakeTable(Path(__file__).parent /  rel_path)
    wake.factor = factor
    wake.step = step

    if w_sampling is not None:
        wake.w_sampling = w_sampling
    if filter_order is not None:
        wake.filter_order = filter_order

    return wake


class A1(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'A1'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'out.ast'
        self.output_beam_file = self.particle_dir + 'section_A1.npz'
        self.tws_file = self.tws_dir + "tws_section_A1.npz"
        # init tracking lattice
        i1_cell = i1.make_cell()

        # Simply the start, just after gun
        a1_start = next(ele for ele in i1_cell if ele.id == "START_SIM")
        # just after last cavity module of A1
        a1_stop = next(ele for ele in i1_cell if ele.id == "a1_sim_stop")

        if "coupler_kick" in kwargs:
            self.coupler_kick = kwargs["coupler_kick"]
        else:
            self.coupler_kick = True

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension

        self.lattice = MagneticLattice(i1_cell, start=a1_start, stop=a1_stop, method=self.method)
        # init physics processes
        sc = make_space_charge(1, [63, 63, 63])
        sc2 = make_space_charge(1, [63, 63, 63])
        wake = make_wake('mod_TESLA_MODULE_WAKE_TAYLOR.dat', 1, 50)
        # adding physics processes
        # just after the first A1 cavity.
        just_after_first_a1_cavity = next(ele for ele in i1_cell if ele.id == "just-after-first-a1-cavity")
        first_cavity = next(ele for ele in i1_cell if ele.id == "C.A1.1.1.I1")
        # beam is immediately smoothed right at the start of the simulation in an instant
        self.add_physics_process(BEAM_SMOOTHER, start=a1_start, stop=a1_start)

        # Attach a SC instance from start (just after gun) to just after first module
        self.add_physics_process(sc, start=a1_start, stop=just_after_first_a1_cavity)
        # Attach a different SC instance between before 2nd module and end of last module.
        self.add_physics_process(sc2, start=just_after_first_a1_cavity, stop=a1_stop)
        # From start of A1 cavities to end of A1 (just after last cavity), attach wake kick.
        self.add_physics_process(wake, start=first_cavity, stop=a1_stop)


class AH1(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'Injector AH1'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'section_A1.npz'
        self.output_beam_file = self.particle_dir + 'section_AH1.npz'
        self.tws_file = self.tws_dir + "tws_section_AH1.npz"
        # init tracking lattice
        i1_cell = i1.make_cell()
        a1_stop = next(ele for ele in i1_cell if ele.id == "a1_sim_stop")
        # This is just before the first laser heater chicane magnet.
        just_before_lh_first_dipole = next(ele for ele in i1_cell if ele.id == "just-before-first-laser-heater-dipole")
        self.lattice = MagneticLattice(i1_cell, start=a1_stop,
                                       stop=just_before_lh_first_dipole,
                                       method=self.method)
        # init physics processes
        sc = make_space_charge(step=5, random_mesh=bRandomMesh)
        wake = make_wake("wake_table_AH1.dat", factor=1, step=10,
                         w_sampling=WakeSampling,
                         filter_order=WakeFilterOrder)

        # adding physics processes
        self.add_physics_process(sc, start=a1_stop, stop=just_before_lh_first_dipole)
        self.add_physics_process(wake, start=i1.c3_ah1_1_1_i1, stop=just_before_lh_first_dipole)


class LH(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'LASER HEATER MAGNETS'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'section_AH1.npz'
        self.output_beam_file = self.particle_dir + 'section_LH.npz'
        self.tws_file = self.tws_dir + "tws_section_LH.npz"
        # init tracking lattice
        acc39_stop = i1.stlat_47_i1
        # lhm_stop = l1.id_90904668_ #for s2e
        # lhm_stop = l1.cix_65_i1    #approx. corresponds to the position of the screen in I1D.
        lhm_stop = i1.dump_csr_start  # for going in I1D
        self.lattice = MagneticLattice(i1.cell + l1.cell, start=acc39_stop, stop=lhm_stop, method=self.method)
        # init physics processes
        csr = CSR()
        csr.sigma_min = Sig_Z[0] * CSRSigmaFactor
        csr.traj_step = 0.0005
        csr.apply_step = 0.005
        sc = SpaceCharge()
        sc.step = 5
        sc.nmesh_xyz = SCmesh
        sc.random_mesh = bRandomMesh
        wake = Wake()
        wake.wake_table = WakeTable('accelerator/wakes/RF/wake_table_TDS1.dat')
        wake.factor = 1
        wake.step = 10
        wake.w_sampling = WakeSampling
        wake.filter_order = WakeFilterOrder

        lh = LaserModulator()
        lh.dE = LHE
        lh.sigma_l = 300
        lh.sigma_x = 300e-6
        lh.sigma_y = 300e-6
        lh.z_waist = None

        filename_tds1 = self.particle_dir + "tds1.npz"
        filename_tds2 = self.particle_dir + "tds2.npz"
        filename_tds3 = self.particle_dir + "tds3.npz"
        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(filename_tds1)
            filename_tds1 = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(filename_tds2)
            filename_tds2 = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(filename_tds3)
            filename_tds3 = filename + str(kwargs["suffix"]) + file_extension

        sv_tds1 = SaveBeam(filename=filename_tds1)
        # sv_tds2 = SaveBeam(filename=filename_tds2)
        # sv_tds3 = SaveBeam(filename=filename_tds3)

        tws_52 = Twiss()
        tws_52.beta_x = 3.131695851
        tws_52.beta_y = 5.417462794
        tws_52.alpha_x = -0.9249364470
        tws_52.alpha_y = 1.730107901
        tws_52.gamma_x = (1 + tws_52.alpha_x**2) / tws_52.beta_x

        tr = BeamTransform(tws=tws_52)
        # tr.bounds = [-0.5, 0.5]
        tr.slice = "Emax"
        self.add_physics_process(sc, start=acc39_stop, stop=lhm_stop)
        self.add_physics_process(csr, start=acc39_stop, stop=lhm_stop)
        self.add_physics_process(wake, start=acc39_stop, stop=lhm_stop)
        self.add_physics_process(sv_tds1, start=i1.tds1, stop=i1.tds1)
        # self.add_physics_process(sv_tds2, start=i1.tds2, stop=i1.tds2)
        # self.add_physics_process(sv_tds3, start=i1.tds3, stop=i1.tds3)
        self.add_physics_process(tr, i1.tmp_m, i1.tmp_m)


class I1D_Screen(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'I1 DUMP'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'section_LH.npz'
        self.output_beam_file = self.particle_dir + 'section_I1D.npz'
        self.tws_file = self.tws_dir + "tws_section_I1D.npz"
        filename_dump = self.particle_dir + "dump.npz"
        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(filename_dump)
            filename_dump = filename + str(kwargs["suffix"]) + file_extension

        sv_dump = SaveBeam(filename=filename_dump)
        # init tracking lattice
        i1d_start = i1.dump_csr_start
        i1d_stop = i1d.otrc_64_i1d
        # i1d_stop = i1d.stsec_62_i1d
        self.lattice = MagneticLattice(i1.cell + i1d.cell, start=i1d_start, stop=i1d_stop, method=self.method)
        # init physics processes
        sigma = Sig_Z[0]
        csr = CSR()
        csr.sigma_min = sigma * 0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005

        sc = SpaceCharge()
        sc.step = 5
        sc.nmesh_xyz = SCmesh
        sc.random_mesh = bRandomMesh

        self.add_physics_process(sc, start=i1d_start, stop=i1d_stop)
        self.add_physics_process(csr, start=i1d_start, stop=i1d.bpma_63_i1d)
        self.add_physics_process(sv_dump, start=i1d_start, stop=i1d_start)


class I1D(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'I1 DUMP'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'section_LH.npz'
        self.output_beam_file = self.particle_dir + 'section_I1D.npz'
        self.tws_file = self.tws_dir + "tws_section_I1D.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            self.input_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension


        # init tracking lattice
        i1d_start = i1.dump_csr_start
        i1d_stop = i1d.ensec_66_i1d
        self.lattice = MagneticLattice(i1.cell + i1d.cell, start=i1d_start, stop=i1d_stop, method=self.method)
        # init physics processes
        sigma = Sig_Z[0]
        csr = CSR()
        csr.sigma_min = sigma * 0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005
        sc = SpaceCharge()
        sc.step = 50
        sc.nmesh_xyz = [63, 63, 63]

        self.add_physics_process(sc, start=i1d_start, stop=i1d_stop)
        self.add_physics_process(csr, start=i1d_start, stop=i1d.bpma_63_i1d)


class DL_New(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'DOGLEG'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'section_LH.npz'
        self.output_beam_file = self.particle_dir + 'section_DL.npz'
        self.tws_file = self.tws_dir + "tws_section_DL.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            self.input_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension

        # init tracking lattice
        st2_stop = i1.dump_csr_start
        dogleg_stop = l1.stlat_96_i1
        self.lattice = MagneticLattice(i1.cell + l1.cell, start=st2_stop, stop=dogleg_stop, method=self.method)
        # init physics processes
        sigma=Sig_Z[0]
        csr = CSR()
        csr.sigma_min = sigma*0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005
        wake_add = Wake()
        wake_add.wake_table = WakeTable('accelerator/wakes/mod_wake_0070.030_0073.450_MONO.dat')
        wake_add.factor = 1

        sc = SpaceCharge()
        sc.step = 25
        sc.nmesh_xyz = [63, 63, 63]
        self.add_physics_process(csr, start=l1.dl_start, stop=dogleg_stop)
        self.add_physics_process(sc, start=st2_stop, stop=dogleg_stop)
        self.add_physics_process(wake_add, start=dogleg_stop, stop=dogleg_stop)


class DL(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        # setting parameters
        self.lattice_name = 'DOGLEG'
        self.unit_step = 0.02
        self.input_beam_file = self.particle_dir + 'section_LH.npz'
        self.output_beam_file = self.particle_dir + 'section_DL.npz'
        self.tws_file = self.tws_dir + "tws_section_DL.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            self.input_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension

        # init tracking lattice
        st2_stop = l1.dl_start
        dogleg_stop = l1.stlat_96_i1
        self.lattice = MagneticLattice(l1.cell, start=st2_stop, stop=dogleg_stop, method=self.method)
        # init physics processes
        sigma=Sig_Z[0]
        csr = CSR()
        csr.sigma_min = sigma*0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005
        wake_add = Wake()
        wake_add.wake_table = WakeTable('accelerator/wakes/mod_wake_0070.030_0073.450_MONO.dat')
        wake_add.factor = 1

        sc = SpaceCharge()
        sc.step = 25
        sc.nmesh_xyz = [63, 63, 63]
        self.add_physics_process(csr, start=st2_stop, stop=dogleg_stop)
        self.add_physics_process(sc, start=st2_stop, stop=dogleg_stop)
        self.add_physics_process(wake_add, start=dogleg_stop, stop=dogleg_stop)


class BC0(SectionTrack):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)

        # setting parameters
        self.lattice_name = 'BC0'
        self.unit_step = 0.1


        self.input_beam_file = self.particle_dir + 'section_DL.npz'
        self.output_beam_file = self.particle_dir + 'section_BC0.npz'
        self.tws_file = self.tws_dir + "tws_section_BC0.npz"


        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            self.input_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension

        # init tracking lattice
        st4_stop = l1.stlat_96_i1
        bc0_stop = l1.enlat_101_i1
        self.lattice = MagneticLattice(l1.cell, start=st4_stop, stop=bc0_stop, method=self.method)

        # init physics processes

        sigma=Sig_Z[0]
        csr = CSR()
        csr.sigma_min = sigma*0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005

        sc = SpaceCharge()
        sc.step = 10
        sc.nmesh_xyz = [63, 63, 63]
        sc.low_order_kick = False
        match_bc0 = st4_stop
        self.add_physics_process(sc, start=match_bc0, stop=bc0_stop)
        self.add_physics_process(csr, start=match_bc0, stop=bc0_stop)
        self.dipoles = [l1.bb_96_i1, l1.bb_98_i1, l1.bb_100_i1, l1.bb_101_i1]
        self.dipole_len = 0.5
        self.bc_gap=1.0

class L1(SectionTrack):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)

        # setting parameters
        self.lattice_name = 'L1'
        self.unit_step = 0.02


        self.input_beam_file = self.particle_dir + 'section_BC0.npz'
        self.output_beam_file = self.particle_dir + 'section_L1.npz'
        self.tws_file = self.tws_dir + "tws_section_L1.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            suff = str(kwargs["suffix"])
            indx = suff.find("_chirpL1_")
            input_suff = suff[:indx]
            self.input_beam_file = filename + input_suff + file_extension
            print("SECTION L1: ", self.input_beam_file)
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension


        bc0_stop = l1.enlat_101_i1
        acc2_stop = l1.stlat_182_b1

        if "coupler_kick" in kwargs:
            self.coupler_kick = kwargs["coupler_kick"]
        else:
            self.coupler_kick = True

        # init tracking lattice
        self.lattice = MagneticLattice(l1.cell, start=bc0_stop, stop=acc2_stop, method=self.method)

        # init physics processes
        smooth = SmoothBeam()
        smooth.mslice = SmoothPar

        sc = SpaceCharge()
        sc.step = 50
        sc.nmesh_xyz = [31, 31, 31]
        wake = Wake()
        wake.wake_table = WakeTable('accelerator/wakes/RF/mod_TESLA_MODULE_WAKE_TAYLOR.dat')
        wake.factor = 4
        wake.step = 100
        wake_add = Wake()
        wake_add.wake_table = WakeTable('accelerator/wakes/mod_wake_0078.970_0159.280_MONO.dat')
        wake_add.factor = 1
        match_acc2 = bc0_stop
        L1_wake_kick = acc2_stop
        self.add_physics_process(smooth, start=match_acc2, stop=match_acc2)
        self.add_physics_process(sc, start=match_acc2, stop=L1_wake_kick)
        self.add_physics_process(wake, start=l1.c_a2_1_1_l1, stop=l1.c_a2_4_8_l1)
        self.add_physics_process(wake_add, start=L1_wake_kick, stop=L1_wake_kick)


class BC1(SectionTrack):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)

        # setting parameters
        self.lattice_name = 'BC1'
        self.unit_step = 0.05

        self.input_beam_file = self.particle_dir + 'section_L1.npz'
        self.output_beam_file = self.particle_dir + 'section_BC1.npz'
        self.tws_file = self.tws_dir + "tws_section_BC1.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            self.input_beam_file = filename + str(kwargs["suffix"]) + file_extension
            print("SECTION B1: ", self.input_beam_file)
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension

        acc2_stop = l1.stlat_182_b1
        bc1_stop = l1.tora_203_b1
        # init tracking lattice
        self.lattice = MagneticLattice(l1.cell, start=acc2_stop, stop=bc1_stop, method=self.method)

        # init physics processes

        sigma = Sig_Z[1]
        csr = CSR()
        #csr.step = 10
        #csr.n_bin = 100
        csr.sigma_min = sigma*0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005

        sc = SpaceCharge()
        sc.step = 20
        sc.nmesh_xyz = [31, 31, 31]
        match_bc1 = acc2_stop
        self.add_physics_process(csr, start=match_bc1, stop=bc1_stop)
        self.add_physics_process(sc, start=match_bc1, stop=bc1_stop)
        self.dipoles = [l1.bb_182_b1, l1.bb_191_b1, l1.bb_193_b1, l1.bb_202_b1]
        self.dipole_len = 0.5
        self.bc_gap=8.5


class L2(SectionTrack):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)

        # setting parameters
        self.lattice_name = 'L2'
        self.unit_step = 0.02

        self.input_beam_file = self.particle_dir + 'section_BC1.npz'
        self.output_beam_file = self.particle_dir + 'section_L2.npz'
        self.tws_file = self.tws_dir + "tws_section_L2.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)
            suff = str(kwargs["suffix"])
            indx = suff.find("_chirpL2_")
            input_suff = suff[:indx]
            self.input_beam_file = filename + input_suff + file_extension
            print("SECTION L2: ", self.input_beam_file)
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension



        bc1_stop = l1.tora_203_b1
        acc3t5_stop = l2.stlat_393_b2

        if "coupler_kick" in kwargs:
            self.coupler_kick = kwargs["coupler_kick"]
        else:
            self.coupler_kick = True

        # init tracking lattice
        self.lattice = MagneticLattice(l1.cell + l2.cell, start=bc1_stop, stop=acc3t5_stop, method=self.method)

        # init physics processes
        smooth = SmoothBeam()
        smooth.mslice = SmoothPar

        sc = SpaceCharge()
        sc.step = 100
        sc.nmesh_xyz = [31, 31, 31]

        wake = Wake()
        wake.wake_table = WakeTable('accelerator/wakes/RF/mod_TESLA_MODULE_WAKE_TAYLOR.dat')
        wake.factor = 4 * 3
        wake.step = 200
        wake_add = Wake()
        wake_add.wake_table = WakeTable('accelerator/wakes/mod_wake_0179.810_0370.840_MONO.dat')
        wake_add.factor = 1
        self.add_physics_process(smooth, start=bc1_stop, stop=bc1_stop)
        self.add_physics_process(sc, start=bc1_stop, stop=acc3t5_stop)
        self.add_physics_process(wake, start=l2.c_a3_1_1_l2, stop=l2.c_a5_4_8_l2)
        self.add_physics_process(wake_add, start=acc3t5_stop, stop=acc3t5_stop)


class BC2(SectionTrack):

    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)

        # setting parameters
        self.lattice_name = 'BC2'
        self.dipoles = [l2.bb_393_b2, l2.bb_402_b2, l2.bb_404_b2, l2.bb_413_b2]
        self.dipole_len = 0.5
        self.bc_gap = 8.5

        self.unit_step = 0.02

        self.input_beam_file = self.particle_dir + 'section_L2.npz'
        self.output_beam_file = self.particle_dir + 'section_BC2.npz'
        self.tws_file = self.tws_dir + "tws_section_BC2.npz"

        if "suffix" in kwargs:
            filename, file_extension = os.path.splitext(self.input_beam_file)

            self.input_beam_file = filename + str(kwargs["suffix"]) + file_extension
            print("SECTION B2: ", self.input_beam_file)
            filename, file_extension = os.path.splitext(self.output_beam_file)
            self.output_beam_file = filename + str(kwargs["suffix"]) + file_extension
            filename, file_extension = os.path.splitext(self.tws_file)
            self.tws_file = filename + str(kwargs["suffix"]) + file_extension

        acc3t5_stop = l2.stlat_393_b2
        bc2_stop = l2.tora_415_b2
        # init tracking lattice
        self.lattice = MagneticLattice(l2.cell, start=acc3t5_stop, stop=bc2_stop, method=self.method)

        # init physics processes

        csr = CSR()
        csr.step=2
        csr.n_bin = 100
        csr.sigma_min = Sig_Z[2]*0.1
        csr.traj_step = 0.0005
        csr.apply_step = 0.005

        sc = SpaceCharge()
        sc.step = 50
        sc.nmesh_xyz = [31, 31, 31]


        self.add_physics_process(csr, start=acc3t5_stop, stop=bc2_stop)
        self.add_physics_process(sc, start=acc3t5_stop, stop=bc2_stop)


class B2D(SectionTrack):
    def __init__(self, data_dir, *args, **kwargs):
        super().__init__(data_dir)
        self.lattice_name = "B2D"

        self.unit_step = 0.02

        self.input_beam_file = self.particle_dir / "section_BC2.npz"
        self.output_beam_file = self.particle_dir / "section_L2.npz"

def make_to_i1d_dump_screen_lattice(data_dir="./"):
    sections = [A1, AH1, LH, I1D_Screen]
    return SectionLattice(sequence=sections, data_dir=data_dir)


def make_to_bc2_dump_screen_lattice():
    all_sections = [A1, AH1, LH, DL, BC0, L1, BC1, L2, BC2, BC2D]  # , L3, CL1, CL2, CL3, STN10]#, SASE1, T4, SASE3, T4D]
    sections = []
    pass

# Need: L1, l2.py
