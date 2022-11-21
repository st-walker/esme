# -*- coding: utf-8 -*-
"""
Sergey Tomin

Script to collect tuning data
"""
import time
from mint.machine import Machine
from mint.snapshot import Snapshot, SnapshotDB


snapshot = Snapshot()
snapshot.sase_sections = []
snapshot.magnet_prefix = None


snapshot.add_alarm_channels("XFEL.DIAG/TOROID/TORA.60.I1/CHARGE.ALL", min=0.005, max=1.5)

snapshot.add_channel("XFEL.UTIL/BUNCH_PATTERN/CONTROL/BEAM_ALLOWED")

snapshot.add_orbit_section("I1", tol=0.1, track=False)
snapshot.add_orbit_section("I1D", tol=0.1, track=False)

snapshot.add_magnet_section("I1", tol=0.01, track=False)
snapshot.add_magnet_section("I1D", tol=0.01, track=False)

# add camera
snapshot.add_image("XFEL.DIAG/CAMERA/OTRC.64.I1D/IMAGE_EXT_ZMQ", folder="./tds_images")
#snapshot.add_image("XFEL.DIAG/CAMERA/OTRA.473.B2D/IMAGE_EXT_ZMQ", folder="./tds_images")
# solenoid
snapshot.add_channel("XFEL.MAGNETS/MAGNET.ML/SOLB.23.I1/CURRENT.SP")

# A1  XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/PHASE.SAMPLE", tol=0.02)
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.A1.I1/AMPL.SAMPLE", tol=0.1)
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.PHASE")
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.A1.I1/SP.AMPL")

# gun phase
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/PHASE.SAMPLE", tol=0.03)
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.PHASE", tol=0.01)
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.GUN.I1/SP.AMPL")
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.GUN.I1/AMPL.SAMPLE")


# AH1
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/PHASE.SAMPLE", tol=0.03)
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.AH1.I1/AMPL.SAMPLE", tol=0.1)
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.PHASE")
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.AH1.I1/SP.AMPL")

# charge

snapshot.add_channel("XFEL.FEEDBACK/FT1.LONGITUDINAL/MONITOR1/TARGET")
snapshot.add_channel("XFEL.DIAG/CHARGE.ML/TORA.25.I1/CHARGE.ALL")

#snapshot.add_channel("XFEL.RF/LINAC_ENERGY_MANAGER/XFEL/ENERGY.2", tol=5)

snapshot.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/LH/ENERGY.ALL", tol=0.2)

snapshot.add_channel("XFEL.DIAG/BEAM_ENERGY_MEASUREMENT/I1D/ENERGY.ALL", tol=0.2)


#BCs
snapshot.add_channel("XFEL.MAGNETS/CHICANE/LH/ANGLE")#mrad
snapshot.add_channel("XFEL.MAGNETS/CHICANE/BC0/ANGLE")
snapshot.add_channel("XFEL.MAGNETS/CHICANE/BC1/ANGLE")
snapshot.add_channel("XFEL.MAGNETS/CHICANE/BC2/ANGLE")

#Laser Heater
snapshot.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1X.LHOS0/FPOS") #Laser position x compare with BPM.48 and .52
snapshot.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/P1Z.LHOS0/FPOS") #Laser position y
snapshot.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/LAMBDA2.LHOS0/POS") #laser intensity, min at 0 max at 7000
snapshot.add_channel("XFEL.UTIL/LASERHEATER.MOTOR/DL.LHLVL5/FPOS") #delay line, on beam at -241
snapshot.add_channel("XFEL.UTIL/LASERINT/GUN/SH3_OPEN") # UG5 shutter open
snapshot.add_channel("XFEL.UTIL/LASERINT/GUN/SH4_OPEN") # UG7 shutter open
# snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSB2/SP.PHASE")
# TDS injector
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/VS.LLTDSI1/AMPL.SAMPLE")
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.AMPL")
snapshot.add_channel("XFEL.RF/LLRF.CONTROLLER/CTRL.LLTDSI1/SP.PHASE")
snapshot.add_channel("XFEL.DIAG/TIMER.CENTRAL/MASTER/EVENT10")  # indication if TDS is on beam


# BAM
#snapshot.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE.RESOLUTION")
#snapshot.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.1")
#snapshot.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.2")
#snapshot.add_channel("XFEL.SDIAG/BAM/47.I1/LOW_CHARGE_SINGLEBUNCH_ARRIVAL_TIME.3")

if __name__ is "__main__":
    def start_collect_data(name = "run_17_"):
        print("Starting collection data ... ")

        machine = Machine(snapshot)
        db_count = 0

        db = SnapshotDB(filename=name + str(db_count) +".p")

        df_ref = machine.get_machine_snapshot()
        if df_ref is not None:

            db.add(df_ref)

        count = 0
        while True:

            #print(".",)
            time.sleep(1)

            df = machine.get_machine_snapshot()
            if df is None:
                continue
            is_diff = snapshot.is_diff(df_ref, df)

            if is_diff:
                df_ref = df
                db.add(df_ref)
                count += 1
                print(count)
            if count % 100 == 1:
                print("saving ...")
                db.save()
            if count % 1000 == 1 and count != 1:
                print("new DB: ", db_count)
                db_count += 1
                db = SnapshotDB(filename=name + str(db_count) +".p")


    #start_collect_data(name="test_")
    machine = Machine(snapshot)
    db = SnapshotDB(filename="test.pcl")
    df_ref = machine.get_machine_snapshot()

    db.add(df_ref)
    time.sleep(1)
    df = machine.get_machine_snapshot()
    is_diff = snapshot.is_diff(df_ref, df)
    db.add(df)
    df_main = db.df

    db.save()
