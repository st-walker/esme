# -*- coding: utf-8 -*-
"""
Sergey Tomin

Script to collect tuning data
"""
import time
from esme.mint.machine import Machine
from esme.mint.snapshot import Snapshot


DUMP_SCREEN_ADDRESS: str = "XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ"

TDS_AMPLITUDE_READBACK_ADDRESS = "XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL"
TDS_AMPLITUDE_SAMPLE_ADDRESS = "XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE"
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


# The hardcoded (in the machine) timing of the first stable bunch
# going into each TDS.  This changes over time.  Bolko updates this
# himself (last time, at time of writing, 2 years ago).  I assume this
# is correct for a given machine configuration
# Each tick here is in units of 9.23 nanoseconds
BUNCH_ONE_TDS_I1 = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSA.52.I1/BUNCH_ONE"
BUNCH_ONE_TDS_BC2 = "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSB.428.B2/BUNCH_ONE"

# "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSA.428.B2/PULSES.ACTIVE"
# "XFEL.SDIAG/SPECIAL_BUNCHES.ML/TDSA.428.B2/PULSES.START"
# "XFEL.SDIAG/SPECIAL_BUNCHES.ML//PULSES.ACTIVE"
# "XFEL.SDIAG/SPECIAL_BUNCHES.ML//PULSES.START"

TDS_ON_BEAM_EVENT10 = [
    1,  # Enabled
    111,  # Some "Event number", all events are numbered.  TDS injector has 111.
    4597792,  # Delay in units of 9.23ns
    2,  # Mask, not sure what this is for...  in any case just change the delay.
]

# 1 tick = 9.23ns.  Need this when adjusting R56.

# time of flight can be a problem.  if i have a crazy time.
# e.g. wrong energy in chicanes, looks like it is coming at the right
# time.  probably only a problem at BC2.

# confirm that the setpoint is matching the rb!  this is a good way of
# checkign the TDS is on!

I1D_SCREEN = "XFEL.DIAG/SCREEN.ML/OTRC.64.I1D/ONAXIS_LYSO"
B2D_SCREEN = "XFEL.DIAG/SCREEN.ML/OTRA.473.B2D/ONAXIS_LYSO"
SCREEN_NOT_IN = 0

SNAPSHOT_TEMPL = Snapshot()
SNAPSHOT_TEMPL.sase_sections = []
SNAPSHOT_TEMPL.magnet_prefix = None


SNAPSHOT_TEMPL.add_alarm_channels("XFEL.DIAG/TOROID/TORA.60.I1/CHARGE.ALL", min=0.005, max=1.5)

SNAPSHOT_TEMPL.add_channel(BEAM_ALLOWED_ADDRESS)

SNAPSHOT_TEMPL.add_orbit_section("I1", tol=0.1, track=False)
SNAPSHOT_TEMPL.add_orbit_section("I1D", tol=0.1, track=False)

SNAPSHOT_TEMPL.add_magnet_section("I1", tol=0.01, track=False)
SNAPSHOT_TEMPL.add_magnet_section("I1D", tol=0.01, track=False)

# add camera

# SNAPSHOT_TEMPL.add_image("XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ", folder="./tds_images")
# solenoid
SNAPSHOT_TEMPL.add_channel("XFEL.MAGNETS/MAGNET.ML/SOLB.23.I1/CURRENT.SP")

# A1  XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE", tol=0.02)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE", tol=0.1)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE")
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL")

# gun phase
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE", tol=0.03)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE", tol=0.01)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.AMPL")
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE")


# AH1
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE", tol=0.03)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE", tol=0.1)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE")
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.AMPL")

# charge

SNAPSHOT_TEMPL.add_channel("XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR1/TARGET")
SNAPSHOT_TEMPL.add_channel("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL")


# SNAPSHOT_TEMPL.add_channel("XFEL.RF/LINAC_ENERGY_MANAGER/XFEL/ENERGY.2", tol=5)

SNAPSHOT_TEMPL.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/LH/ENERGY.ALL", tol=0.2)

SNAPSHOT_TEMPL.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", tol=0.2)


# BCs
SNAPSHOT_TEMPL.add_channel("XFEL.MAGNETS/CHICANE/LH/ANGLE")  # mrad
SNAPSHOT_TEMPL.add_channel("XFEL.MAGNETS/CHICANE/BC0/ANGLE")
SNAPSHOT_TEMPL.add_channel("XFEL.MAGNETS/CHICANE/BC1/ANGLE")
SNAPSHOT_TEMPL.add_channel("XFEL.MAGNETS/CHICANE/BC2/ANGLE")

# Laser Heater
SNAPSHOT_TEMPL.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1X.LHOS0/FPOS")  # Laser position x compare with BPM.48 and .52
SNAPSHOT_TEMPL.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1Z.LHOS0/FPOS")  # Laser position y
SNAPSHOT_TEMPL.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/LAMBDA2.LHOS0/POS")  # laser intensity, min at 0 max at 7000
SNAPSHOT_TEMPL.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/DL.LHLVL5/FPOS")  # delay line, on beam at -241
SNAPSHOT_TEMPL.add_channel("XFEL.UTIL/LASERINT/GUN/SH3_OPEN")  # UG5 shutter open
SNAPSHOT_TEMPL.add_channel("XFEL.UTIL/LASERINT/GUN/SH4_OPEN")  # UG7 shutter open
# SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/SP.PHASE")

# TDS injector
SNAPSHOT_TEMPL.add_channel(TDS_AMPLITUDE_SAMPLE_ADDRESS)
SNAPSHOT_TEMPL.add_channel(TDS_AMPLITUDE_READBACK_ADDRESS)
SNAPSHOT_TEMPL.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.PHASE")
SNAPSHOT_TEMPL.add_channel(EVENT10_CHANNEL)  # indication if TDS is on beam


# BAM
# SNAPSHOT_TEMPL.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE.RESOLUTION")
# SNAPSHOT_TEMPL.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.1")
# SNAPSHOT_TEMPL.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.2")
# SNAPSHOT_TEMPL.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.3")
