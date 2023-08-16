import yaml
import os
from typing import TextIO, Any


from esme.control.kickers import FastKicker, FastKickerController, PolarityType, FastKickerSetpoint
from esme.control.screens import Screen, ScreenService
from esme.control.machines import BunchLengthMachine
from esme.control.tds import TransverseDeflector, TransverseDeflectors
from esme.control.sbunches import DiagnosticRegion


def load_kickers_from_config(dconf: dict[str, Any]) -> FastKickerController:
    kicker_defs = dconf["kickers"]
    kickers = []
    for kdef in kicker_defs:
        kickers.append(FastKicker(name=kdef["name"],
                                  adio24_fdl=kdef["adio24_fdl"],
                                  trigger_channel=kdef["trigger_channel"]))
                                  
    return FastKickerController(kickers)

def load_screens_from_config(dconf: dict[str, Any]) -> ScreenService:
    screen_defs = dconf["screens"]
    screens = []
    for sdef in screen_defs:
        screen_name = sdef["name"]
        location = DiagnosticRegion(sdef["area"])
        try:
            kicker_sp_defs = sdef["kickers"]
        except KeyError:
            kicker_setpoints = None
        else:
            kicker_setpoints = []
            for kicker_name, kicker_sp_def in kicker_sp_defs.items():
                ksp = FastKickerSetpoint(name=kicker_name,
                                         voltage=kicker_sp_def["voltage"],
                                         delay=kicker_sp_def["delay"],
                                         polarity=parse_polarity(kicker_sp_def))
                kicker_setpoints.append(ksp)

        screens.append(Screen(name=screen_name,
                              location=location,
                              fast_kicker_setpoints=kicker_setpoints))

    return ScreenService(screens)

def parse_polarity(cdict):
    try:
        return PolarityType(cdict["polarity"])
    except KeyError:
        return None
    

def build_simple_machine_from_config(yamlf: os.PathLike) -> BunchLengthMachine:
    with open(yamlf, "r") as f:
        config = yaml.full_load(f)

    kickercontroller = load_kickers_from_config(config)
    screenservice = load_screens_from_config(config)
    deflectors = load_deflectors_from_config(config)

    return BunchLengthMachine(kickercontroller, screenservice, deflectors)
    

    
def load_deflectors_from_config(dconf: dict[str, Any]) -> TransverseDeflectors:
    deflector_defs = dconf["deflectors"]

    deflectors = []
    for ddef in deflector_defs:
        area = DiagnosticRegion(ddef["area"])
        sp_fdl = ddef["sp_fdl"]
        rb_fdl = ddef["rb_fdl"]

        deflector = TransverseDeflector(area, sp_fdl, rb_fdl)
        deflectors.append(deflector)

    return TransverseDeflectors(deflectors)

