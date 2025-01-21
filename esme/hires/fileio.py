import json
import typing

from .data import ImageAnalysisResult


def load_analysis_result(fp: str | typing.IO) -> ImageAnalysisResult:
    try:
        with open(fp, "r") as f:
            anadict = json.load(f)
    except TypeError:
        anadict = json.load(fp)

    samples = anadict["Samples"]["Analysis"]

    # for sample in samples:

    # from IPython import embed; embed()
