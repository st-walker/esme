# -*- coding: utf-8 -*-
"""
Sergey Tomin

Script to collect tuning data
"""

import os

from esme.mint import Snapshot, BasicAlarm


DUMP_SCREEN_ADDRESS: str = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"
I1D_SCREEN_ADDRESS = DUMP_SCREEN_ADDRESS
B2D_SCREEN_ADDRESS = "NOTHING EYET"

TDS_I1_AMPLITUDE_READBACK_ADDRESS = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL"
TDS_I1_AMPLITUDE_SAMPLE_ADDRESS = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE"
TDS_B2_AMPLITUDE_READBACK_ADDRESS = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/SP.AMPL"
TDS_B2_AMPLITUDE_SAMPLE_ADDRESS = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSB2/AMPL.SAMPLE"


BEAM_ALLOWED_ADDRESS = "XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED"
BEAM_ENERGY_ADDRESS = "XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL"

# Everything here assumes only one bunch in the machine.

# (See: XFEL.DIAG/TIMER.CENTRAL/MASTER/SASE (SlaveTiming.xml))
# 10 because the injector TDS event is numbered 10
# 12 because the BC2 TDS event is numbered 12.
EVENT10_CHANNEL = "XFEL.DIAG/TIMER.CENTRAL/MASTER/EVENT10"
EVENT12_CHANNEL = "XFEL.DIAG/TIMER.CENTRAL/MASTER/EVENT12"

# Only edit the third entry to the arrays of the above EVENT10 or
# EVENT 12 should be edited Either they are off or on-beam.  On beam
# should be switched to BUNCH_ONE, off beam should be whatever it was
# before (so just toggle between the two).

# timing of the first stable bunch
# going into each TDS.  This changes over time.  Bolko updates this
# himself (last time, at time of writing, 2 years ago).  I assume this
# is correct for a given machine configuration
# Each tick here is in units of 9.23 nanoseconds
BUNCH_ONE_TDS_I1 = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSA.52.I1/BUNCH_ONE"   # 4_597_792
BUNCH_ONE_TDS_B2 = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSB.428.B2/BUNCH_ONE" # 6_540_490



BUNCH_ONE_TOLERANCE = 0.05

PULSES_ACTIVE = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSB.428.B2/PULSES.ACTIVE"

off_beam_time = 4_592_892

# "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSA.428.B2/PULSES.ACTIVE"
# "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSA.428.B2/PULSES.START"
# "XFEL.SDIAG/SPECIAL_BUNCHES.ML//PULSES.ACTIVE"
# "XFEL.SDIAG/SPECIAL_BUNCHES.ML//PULSES.START"

TDS_I1_ON_BEAM_EVENT10 = [
    1,  # Enabled
    111,  # Some "Event number", all events are numbered.  TDS injector has 111.
    4_597_792,  # Delay in units of 9.23ns
    2,  # Mask, not sure what this is for...  in any case just change the delay.
]

# 1 tick = 9.23ns.  Need this when adjusting R56.

# time of flight can be a problem.  if i have a crazy time.
# e.g. wrong energy in chicanes, looks like it is coming at the right
# time.  probably only a problem at BC2...
# might need new r56

# confirm that the setpoint is matching the rb!  this is a good way of
# checkign the TDS is on!

I1D_SCREEN = "XFEL.DIAG/SCREEN.ML/OTRC.64.I1D/ONAXIS_LYSO"
B2D_SCREEN = "XFEL.DIAG/SCREEN.ML/OTRA.473.B2D/ONAXIS_LYSO"
SCREEN_NOT_IN = 0


def make_injector_snapshot_template(outdir: os.PathLike):
    template = make_common_snapshot_template()
    _add_injector_to_template(template, outdir)
    return template


def make_b2d_snapshot_template(outdir: os.PathLike):
    template = make_common_snapshot_template()
    _add_injector_to_template(template, outdir)
    _add_b2_snapshot_to_template(template, outdir)    
    return template


def make_common_snapshot_template():
    template = Snapshot()
    # Beam on/off
    template.add_channel(BEAM_ALLOWED_ADDRESS)

    # Orbit
    template.add_orbit_section("I1", tol=0.1)

    # Magnet settings
    template.add_magnet_section("I1", tol=0.01)

    # Beam on/off
    template.add_channel(BEAM_ALLOWED_ADDRESS)
    
    # add camera
    # template.add_image("XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ", folder="./tds_images")
    # solenoid
    template.add_channel("XFEL.MAGNETS/MAGNET.ML/SOLB.23.I1/CURRENT.SP")
    
    # A1
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL")

    # Gun
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE", tol=0.03)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE", tol=0.01)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.AMPL")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE")

    # AH1
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE", tol=0.03)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.AMPL")

    # charge
    template.add_channel("XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR1/TARGET")
    template.add_channel("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL")

    # Energy
    template.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/LH/ENERGY.ALL", tol=0.2)
    template.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", tol=0.2)

    # BC settings
    template.add_channel("XFEL.MAGNETS/CHICANE/LH/ANGLE")  # mrad
    template.add_channel("XFEL.MAGNETS/CHICANE/BC0/ANGLE")
    template.add_channel("XFEL.MAGNETS/CHICANE/BC1/ANGLE")
    template.add_channel("XFEL.MAGNETS/CHICANE/BC2/ANGLE")

    # Laser Heater
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1X.LHOS0/FPOS")  # Laser position x compare with BPM.48 and .52
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1Z.LHOS0/FPOS")  # Laser position y
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/LAMBDA2.LHOS0/POS")  # laser intensity, min at 0 max at 7000
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/DL.LHLVL5/FPOS")  # delay line, on beam at -241
    template.add_channel("XFEL.UTIL/LASERINT/GUN/SH3_OPEN")  # UG5 shutter open
    template.add_channel("XFEL.UTIL/LASERINT/GUN/SH4_OPEN")  # UG7 shutter open
    # template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/SP.PHASE")


def _add_injector_to_template(template, outdir):
    
    screen_name = Path("XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ").parent.name
    image_dir = Path(outdir) / f"images-{screen_name}"
    template.add_image(DUMP_SCREEN_ADDRESS, folder=image_dir)

    # Alarms
    template.alarms.append(BasicAlarm("XFEL.DIAG/TOROID/TORA.60.I1/CHARGE.ALL",
                                      vmin=0.005, message="Charge too small, no beam?"))

    # template.alarms.append(BinaryOpAlarm(), 

    # Orbit
    template.add_orbit_section("I1", tol=0.1)
    template.add_orbit_section("I1D", tol=0.1)

    # Magnet settings
    template.add_magnet_section("I1", tol=0.01)
    template.add_magnet_section("I1D", tol=0.01)

    # add camera
    # template.add_image("XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ", folder="./tds_images")
    # solenoid

    # A1
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL")

    # Gun
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE", tol=0.03)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE", tol=0.01)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.AMPL")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE")

    # AH1
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE", tol=0.03)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.AMPL")

    # charge
    template.add_channel("XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR1/TARGET")
    template.add_channel("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL")

    # Energy
    template.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/LH/ENERGY.ALL", tol=0.2)
    template.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", tol=0.2)

    # BC settings
    template.add_channel("XFEL.MAGNETS/CHICANE/LH/ANGLE") # mrad
    template.add_channel("XFEL.MAGNETS/CHICANE/BC0/ANGLE")
    template.add_channel("XFEL.MAGNETS/CHICANE/BC1/ANGLE")
    template.add_channel("XFEL.MAGNETS/CHICANE/BC2/ANGLE")

    # Laser Heater
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1X.LHOS0/FPOS")  # Laser position x compare with BPM.48 and .52
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1Z.LHOS0/FPOS")  # Laser position y
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/LAMBDA2.LHOS0/POS")  # laser intensity, min at 0 max at 7000
    template.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/DL.LHLVL5/FPOS")  # delay line, on beam at -241
    template.add_channel("XFEL.UTIL/LASERINT/GUN/SH3_OPEN")  # UG5 shutter open
    template.add_channel("XFEL.UTIL/LASERINT/GUN/SH4_OPEN")  # UG7 shutter open
    # template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/SP.PHASE")

    # TDS injector
    template.add_channel(TDS_AMPLITUDE_SAMPLE_ADDRESS)
    template.add_channel(TDS_AMPLITUDE_READBACK_ADDRESS)
    template.add_channel(EVENT10_CHANNEL) # TDS timing (returns 4-tuple)
    template.add_channel(BUNCH_ONE_TDS_I1) # Timing of TDS for first on-beam bunch.

def _add_bc2_channels_to_template(template, outdir):
    screen_name = Path("XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ").parent.name
    image_dir = Path(outdir) / f"images-{screen_name}"
    template.add_image(DUMP_SCREEN_ADDRESS, folder=image_dir)
    
    # Magnets
    template.add_magnet_section("B1", tol=0.01)
    template.add_magnet_section("L1", tol=0.01)
    template.add_magnet_section("L2", tol=0.01)
    template.add_magnet_section("B2", tol=0.01)
    template.add_magnet_section("B2D", tol=0.01)

    # Orbit
    template.add_orbit_section("B1", tol=0.01)
    template.add_orbit_section("L1", tol=0.01)
    template.add_orbit_section("L2", tol=0.01)
    template.add_orbit_section("B2", tol=0.01)
    template.add_orbit_section("B2D", tol=0.01)

    
    # L1
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL")
    
    # L2
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A2.L1/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A2.L1/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A2.L1/SP.AMPL")

    # A3.L2
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A3.L2/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A3.L2/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A3.L2/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A3.L2/SP.AMPL")

    # A4.L2
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A4.L2/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A4.L2/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A4.L2/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A4.L2/SP.AMPL")

    # A5.L2
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A5.L2/PHASE.SAMPLE", tol=0.02)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A5.L2/AMPL.SAMPLE", tol=0.1)
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A5.L2/SP.PHASE")
    template.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A5.L2/SP.AMPL")

    # TDS B2
    template.add_channel(TDS_B2_AMPLITUDE_SAMPLE_ADDRESS)
    template.add_channel(TDS_B2_AMPLITUDE_READBACK_ADDRESS)
    template.add_channel(EVENT12_CHANNEL) # TDS timing (returns 4-tuple)
    template.add_channel(BUNCH_ONE_TDS_B2) # Timing of TDS for first on-beam bunch.
    
