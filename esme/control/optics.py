import pandas as pd
from oxfel.predefined import cat_to_i1d, cat_to_b2d
from esme.control.snapshot import SnapshotRequest, Snapshotter
from esme.optics import (calculate_i1d_r34_from_tds_centre, 
                         i1d_conf_from_measurement_df, 
                         dispersions_at_point,
                         SliceEmittanceMeasurement,
                         track_slice_twiss,
                         calculate_design_i1_r34_from_tds_centre)
from .dint import DOOCSInterface, DOOCSInterfaceABC
from esme.control.exceptions import DOOCSReadError


# XXX: This will not work if just doing an off axis measurement in the
# injector with beam going straight, for example.
# Tried in order:  Why?  if beamline is readable then we are there.
INJECTOR_ENERGY_ADDRESSES = ["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL",
                             "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/B1/ENERGY.ALL"]

INJECTOR_WILDCARDS = ["XFEL.MAGNETS/MAGNET.ML/*.I1/KICK_MRAD.SP",
                      "XFEL.MAGNETS/MAGNET.ML/*.I1D/KICK_MRAD.SP"]

INJECTOR_ADDRESSES = ["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL",
                      "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE",
                      "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE",
                      "XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE",
                      "XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE"]

B2_ENERGY = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/B2D/ENERGY.ALL"

B2_WILDCARDS = ["XFEL.MAGNETS/MAGNET.ML/*.I1/KICK_MRAD.SP",
                 "XFEL.MAGNETS/MAGNET.ML/*.I1D/KICK_MRAD.SP"]

B2_ADDRESSES = ["XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL",
                "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE",
                "XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE",
                "XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE",
                "XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE"]


def make_i1_snapshot_request():
    return SnapshotRequest(addresses=INJECTOR_ADDRESSES,
                           wildcards=INJECTOR_WILDCARDS,
                           image=None)


def make_b2_snapshot_request():
    return SnapshotRequest(addresses=B2_ADDRESSES,
                           wildcards=B2_WILDCARDS,
                           image=None)


class MachineLinearOptics:
    def __init__(self, snapshotter, di: DOOCSInterfaceABC | None = None):
        self.snapshotter = snapshotter
        self.di = di if di else DOOCSInterface()


class I1toI1DLinearOptics(MachineLinearOptics):
    def __init__(self,  di: DOOCSInterfaceABC | None = None):
        snapshotter = Snapshotter(make_i1_snapshot_request(), di=di)
        super().__init__(snapshotter, di=di)
        self.felmodel = cat_to_i1d(model_type="real")

    def r12_streaking_from_tds_to_point(self, screen_or_marker_name: str) -> float:
        # Strictly this is r34 not r12, but point is in streaking plane...
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        i1d_conf_from_measurement_df(df)
        return calculate_i1d_r34_from_tds_centre(df, screen_or_marker_name, self.get_beam_energy())

    def design_r12_streaking_from_tds_to_point(self, screen_or_marker_name: str) -> float:
        return calculate_design_i1_r34_from_tds_centre(screen_or_marker_name)

    def dispersions_at_screen(self, screen_or_marker_name: str) -> tuple[float, float]:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = i1d_conf_from_measurement_df(df)
        return dispersions_at_point(self.felmodel, felconfig, screen_or_marker_name)

    def get_beam_energy(self) -> float:
        """returns beam energy in I1D in MeV"""
        for address in INJECTOR_ENERGY_ADDRESSES:
            try:
                return self.di.get_value(address)
            except DOOCSReadError:
                continue

        raise DOOCSReadError("Unable to read injector beam energy")

    def track_measured_slice_twiss(self, start: str, stop: str, stwiss0: SliceEmittanceMeasurement) -> pd.DataFrame:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = i1d_conf_from_measurement_df(df)
        return track_slice_twiss(self.felmodel, felconfig, start=start, stop=stop, stwiss0=stwiss0)


class I1toB2DLinearOptics(MachineLinearOptics):
    def __init__(self, di: DOOCSInterfaceABC | None = None) -> None:
        snapshotter = Snapshotter(make_b2_snapshot_request(), di=di)
        super().__init__(snapshotter, di=di)
        self.felmodel = cat_to_b2d(model_type="real")

    def r12_streaking_from_tds_to_point(self, screen_or_marker_name: str) -> float:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        i1d_conf_from_measurement_df(df)
        return calculate_b2d_r12_from_tds_centre(df,
                                                 screen_or_marker_name,
                                                 self.get_beam_energy())
