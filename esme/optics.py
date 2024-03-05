from scipy.io import loadmat
import pandas as pd
import numpy as np
from oxfel.predefined import cat_to_i1d
from oxfel.fel_track import EuXFELSimConfig
from oxfel.optics import get_default_match_point
from ocelot.cpbd.elements import Cavity

from ocelot.cpbd.optics import Twiss, twiss as oce_calc_twiss
from ocelot.cpbd.magnetic_lattice import MagneticLattice

from ocelot.cpbd.beam import Twiss



class SliceEmittanceMeasurement:
    def __init__(self, emittance, alpha, beta, alpha_err=None, beta_err=None, emit_err=None):
        self.emittance = emittance
        self.alpha = alpha
        self.beta = beta
        self.alpha_err = alpha_err
        self.beta_err = beta_err
        self.emit_err = emit_err


        
    @property
    def nslices(self):
        return len(self.beta)


def load_matthias_slice_measurement(fname: str) -> SliceEmittanceMeasurement:
    mat = loadmat(fname, squeeze_me=True, simplify_cells=True)
    slice_twiss = mat["result"]["fit_x"]["Gaussian"]

    beta_x = slice_twiss["beta"]
    beta_x_err = slice_twiss["beta_err"]

    alpha_x = slice_twiss["alpha"]
    alpha_x_err = slice_twiss["alpha_err"]

    emit = slice_twiss["emittace_norm"] # Yes emittace not emittance...
    emit_err = slice_twiss["emittace_norm_err"] # Yes emittace not emittance...

    return SliceEmittanceMeasurement(emittance=emit,
                                     alpha=alpha_x,
                                     beta=beta_x,
                                     alpha_err=alpha_x_err,
                                     beta_err=beta_x_err,
                                     emit_err=emit_err)


def i1d_conf_from_measurement_df(df):
    quad_names = df.keys()[df.keys().str.match("^Q")]
    dipole_names = df.keys()[df.keys().str.match("^B[LB]\.")]

    k1ls = df.iloc[0][quad_names] # mrad
    angles = df.iloc[0][dipole_names] # mrad

    n_cavities_a1 = 8
    a1v = df["XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE"].iloc[0] / n_cavities_a1
    df["XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE"]

    conf = EuXFELSimConfig()

    r56 = -4.336e-3
    conf.controls["lh"].r56 = r56

    dump_angle = np.radians(-30)
    dipoles = {"BB.62.I1D": {"angle": dump_angle, "e1": dump_angle / 2, "e2": dump_angle / 2}}
    quads = {quad_name: {"k1l": float(k1l)*1e-3} for quad_name, k1l in k1ls.items()}

    conf.components = quads | dipoles
    conf.controls["ah1"].active = False
    conf.controls["a1"].v = a1v * 1e-3
    conf.controls["a1"].phi = 0

    return conf


def optics_from_measurement_df(df):
    lat = cat_to_i1d(model_type="real")

    conf = i1d_conf_from_measurement_df(df)

    twiss52 = get_default_match_point("MATCH.52.I1")
    twiss52.E = 0.13
    
    twiss_after, mlat = lat.calculate_twiss(twiss52, start="MATCH.52.I1", felconfig=conf)

    before_match52 = lat.get_sequence(stop="MATCH.52.I1", felconfig=conf)
    before_match52.reverse()

    for element in before_match52:
        if isinstance(element, Cavity):
            print(element.phi)
            element.phi += 180

    # Prepare initial twiss for bactracking
    twiss52.s = 0
    twiss52.alpha_x *= -1
    twiss52.alpha_y *= -1

    # chicane = before_match52.closed_interval("BL.50II.I1", "BL.48I.I1")
    # mlat = MagneticLattice(chicane)
    # mlat.transfer_maps(energy=130e-3)
    # print(m[4, 5] * 1e3)


    mlat = MagneticLattice(before_match52)
    twiss_before = oce_calc_twiss(mlat, twiss52, return_df=True)

    # Rearrange the backtracked twiss and get the right value of s forwards
    twiss_before = twiss_before.iloc[::-1]
    s_before = np.array(twiss_before.s)
    s_before = s_before[0] - s_before
    twiss_before["s"] = s_before

    full_twiss = pd.concat([twiss_before, twiss_after])

    return full_twiss, lat.get_sequence()

# def calculate_i1d_r34_from_tds_centre(setpoint):
#     fel = cat_to_i1d(model_type="real")
#     conf = i1d_conf_from_measurement_df(setpoint.df)

#     sequence = fel.get_sequence(start="TDSA.52.I1",
#                                 stop=setpoint.screen_name,
#                                 felconfig=conf)
#     # Halve the TDS length so we are starting half way through it.
#     sequence[0].l /= 2.0
#     mlat = MagneticLattice(sequence)
#     _, rmat, _ = mlat.transfer_maps(setpoint.energy * 1e-3) # MeV to GeV
#     r34 = rmat[2, 3]
#     return r34


def calculate_i1d_r34_from_tds_centre(df, screen_name, energy_mev):
    fel = cat_to_i1d(model_type="real")
    conf = i1d_conf_from_measurement_df(df)

    sequence = fel.get_sequence(start="TDSA.52.I1",
                                stop=screen_name,
                                felconfig=conf)
    # Halve the TDS length so we are starting half way through it.
    sequence[0].l /= 2.0
    mlat = MagneticLattice(sequence)
    _, rmat, _ = mlat.transfer_maps(energy_mev * 1e-3) # MeV to GeV
    r34 = rmat[2, 3]
    return r34

def dispersions_at_point(fel, felconfig, screen_name):
    twiss, mlat = fel.machine_twiss(stop=screen_name, felconfig=felconfig)
    end = twiss.iloc[-1]

    import pickle
    with open("mlat.pkl", "wb") as f:
        pickle.dump(mlat, f)

    with open("fel.pkl", "wb") as f:
        pickle.dump(fel, f)

    with open("felconfig.pkl", "wb") as f:
        pickle.dump(felconfig, f)


    return end.Dx, end.Dy



def track_slice_twiss(fel, felconfig, start: str, stop: str, stwiss0):
    all_slices = []
    for alpha, beta in zip(stwiss0.alpha, stwiss0.beta):
        # Have to include alpha y and beta y otherwise ocelot
        # implicitly tries to find periodic solution...
        twiss = Twiss(alpha_x=alpha, beta_x=beta, beta_y=1, alpha_y=1)
        result, _ = fel.calculate_twiss(twiss, start=start, stop=stop, felconfig=felconfig)

        all_slices.append(result.iloc[-1])

    df = pd.DataFrame(all_slices)
    del df["beta_y"]
    del df["alpha_y"]
    del df["gamma_y"]

    return df
