"""Module for representing the data for a high resolution energy spread measurement"""

from dataclasses import dataclass

from esme.core import DiagnosticRegion


@dataclass
class TDSProfile:
    region: DiagnosticRegion
    length: float
    twiss_alpha: float
    twiss_beta: float
    frequency: float
    # calibration: ???


@dataclass
class ScreenProfile:
    name: str
    energy: float
    dispersion: float | tuple[float, float]
    px_xsize: float
    px_ysize: float


@dataclass
class ScanSetPoint:
    variable: float | tuple[float, float]
    image_analysis: ImageAnalysisResult
    images: list


@dataclass
class GaussFit:
    amplitude: float
    amplitude_err: float
    mean: float
    mean_err: float
    sigma: float
    sigma_err: float


@dataclass
class SliceAnalysisResult:
    time_index: list[int]
    profile: list[float]
    gaussfit: GaussFit


@dataclass
class ImageAnalysisResult:
    samples: SliceAnalysisResult
    result: SliceAnalysisResult


class HREnergySpreadMeasurement:
    def __init__(self, tds: TDSProfile, screen: ScreenProfile):
        pass
