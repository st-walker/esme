

@dataclass
class QuadrupoleSetting:
    """Class for storing a single strength configuration of the
    quadrupole scan, as well as the intended dispersion it is supposed
    to achieve at the screen.

    The integrated strengths and are assumed throughout this package to have units of m^{-1}.

    """

    names: list
    k1ls: list
    dispersion: float

    def __post_init__(self):  # TODO: test this!
        if len(self.names) != len(self.k1ls):
            raise ValueError("Length mismatch between names and integrated strengths")

    def k1l_from_name(self, name):
        index = self.names.index(name)
        return self.k1ls[index]

@dataclass
class DispersionScanConfiguration:
    """A series of QuadrupoleSetting instances for each datapoint in
    the dispersion scan, as well as the "reference_setting"---used in
    the TDS scan and set at the start of the dispersion scan."""

    reference_setting: QuadrupoleSetting
    scan_settings: list[QuadrupoleSetting]
    tds_voltage: float

    @property
    def dispersions(self) -> list[float]:
        return [qsetting.dispersion for qsetting in self.scan_settings]
    


class Beamline:
    def __init__(self):
        pass
        
