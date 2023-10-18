import pandas as pd
import numpy as np
from oxfel.predefined import cat_to_i1d
from esme.control.snapshot import SnapshotRequest, Snapshotter
from esme.optics import calculate_i1d_r34_from_tds_centre, i1d_conf_from_measurement_df

INJECTOR_ENERGY = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

INJECTOR_WILDCARDS = ["XFEL.MAGNETS/MAGNET.ML/*.I1/KICK_MRAD.SP",
                      "XFEL.MAGNETS/MAGNET.ML/*.I1D/KICK_MRAD.SP"]
INJECTOR_ADDRESSES = ["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL",
                      "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE",
                      "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE",
                      "XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE",
                      "XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE"]



def make_snapshot_request():
    return SnapshotRequest(addresses=INJECTOR_ADDRESSES,
                           wildcards=INJECTOR_WILDCARDS,
                           image=None)


class I1toI1DSequenceOptics:
    def __init__(self, mi=None):
        self.mi = mi if mi else XFELMachineInterface()
        self.snapshotter = Snapshotter(make_snapshot_request(), mi=mi)

    def r34_from_tds_to_point(self, screen_or_marker_name):
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = i1d_conf_from_measurement_df(df)
        return calculate_i1d_r34_from_tds_centre(df,
                                                 screen_or_marker_name,
                                                 self.get_dumpline_beam_energy())

    def get_dumpline_beam_energy(self) -> float:
        """returns beam energy in I1D in MeV"""
        energy = self.mi.get_value(INJECTOR_ENERGY)
        return energy
