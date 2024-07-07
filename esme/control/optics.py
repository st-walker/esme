from abc import ABC, abstractmethod

import pandas as pd
from oxfel.predefined import cat_to_b2d, cat_to_i1d

from esme.control.exceptions import DOOCSReadError
from esme.control.snapshot import SnapshotRequest, Snapshotter
from esme.optics import (
    SliceEmittanceMeasurement,
    calculate_design_i1_r34_from_tds_centre,
    calculate_i1d_r34_from_tds_centre,
    dispersions_at_point,
    i1d_conf_from_measurement_df,
    track_slice_twiss,
)

from .dint import DOOCSInterface, DOOCSInterfaceABC


class MachineLinearOptics(ABC):
    def __init__(self, snapshotter, di: DOOCSInterfaceABC | None = None):
        self.snapshotter = snapshotter
        self.di = di if di else DOOCSInterface()

    def full_read(self):
        pass

    def get_beam_energy(self) -> float:
        """returns beam energy in section in MeV"""
        for address in self.energy_addresses:
            try:
                return self.di.get_value(address)
            except DOOCSReadError:
                continue

        raise DOOCSReadError("Unable to read injector beam energy")

    @abstractmethod
    def r12_streaking_from_tds_to_point(self, point_name: str) -> float:
        pass

    @abstractmethod
    def dispersions_at_screen(self, point_name: str) -> tuple[float, float]:
        pass

    def optics_snapshot(self) -> pd.DataFrame:
        return pd.DataFrame.from_records([self.snapshotter.snapshot()])


class I1toI1DLinearOptics(MachineLinearOptics):
    def __init__(
        self,
        request: SnapshotRequest,
        energy_addresses: list[str],
        di: DOOCSInterfaceABC | None = None,
    ):
        snapshotter = Snapshotter(request, di=di)
        super().__init__(snapshotter, di=di)
        self.energy_addresses = energy_addresses
        self.felmodel = cat_to_i1d(model_type="real")

    def r12_streaking_from_tds_to_point_from_df(self, optics_df: pd.DataFrame, target_name: str, beam_energy: float) -> float:
        return calculate_i1d_r34_from_tds_centre(
            optics_df, target_name, beam_energy
        )
        
    def r12_streaking_from_tds_to_point(self, screen_or_marker_name: str) -> float:
        # Strictly this is r34 not r12, but point is in streaking plane...
        return self.r12_streaking_from_tds_to_point_from_df(self.optics_snapshot(),
                                                            screen_or_marker_name,
                                                            self.get_beam_energy())

    def design_r12_streaking_from_tds_to_point(
        self, screen_or_marker_name: str
    ) -> float:
        return calculate_design_i1_r34_from_tds_centre(screen_or_marker_name)

    def dispersions_at_screen(self, screen_or_marker_name: str) -> tuple[float, float]:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = i1d_conf_from_measurement_df(df)
        return dispersions_at_point(self.felmodel, felconfig, screen_or_marker_name)

    def track_measured_slice_twiss(
        self, start: str, stop: str, stwiss0: SliceEmittanceMeasurement
    ) -> pd.DataFrame:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = i1d_conf_from_measurement_df(df)
        return track_slice_twiss(
            self.felmodel, felconfig, start=start, stop=stop, stwiss0=stwiss0
        )

    def full_read(self):
        return self.snapshotter.snapshot(resolve_wildcards=True)


class I1toB2DLinearOptics(MachineLinearOptics):
    def __init__(
        self,
        request: SnapshotRequest,
        energy_addresses: list[str],
        di: DOOCSInterfaceABC | None = None,
    ):
        snapshotter = Snapshotter(request, di=di)
        super().__init__(snapshotter, di=di)
        self.energy_addresses = energy_addresses
        self.felmodel = cat_to_b2d(model_type="real")

    def r12_streaking_from_tds_to_point(self, screen_or_marker_name: str) -> float:
        # Strictly this is r34 not r12, but point is in streaking plane...
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        return calculate_b2d_r12_from_tds_centre(
            df, screen_or_marker_name, self.get_beam_energy()
        )

    def design_r12_streaking_from_tds_to_point(
        self, screen_or_marker_name: str
    ) -> float:
        return calculate_design_b2_r12_from_tds_centre(screen_or_marker_name)

    def dispersions_at_screen(self, screen_or_marker_name: str) -> tuple[float, float]:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = b2d_conf_from_measurement_df(df)
        return dispersions_at_point(self.felmodel, felconfig, screen_or_marker_name)

    def track_measured_slice_twiss(
        self, start: str, stop: str, stwiss0: SliceEmittanceMeasurement
    ) -> pd.DataFrame:
        df = pd.DataFrame.from_records([self.snapshotter.snapshot()])
        felconfig = b2d_conf_from_measurement_df(df)
        return track_slice_twiss(
            self.felmodel, felconfig, start=start, stop=stop, stwiss0=stwiss0
        )

    def full_read(self):
        return self.snapshotter.snapshot(resolve_wildcards=True)
