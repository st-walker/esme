
---
TDS stuff:
---


how to turn TDS on (in bc2):

go to overview:

set:
charge voltage: 1150


assuming klystron is in standby (meaning filament is already heated):

climb ladder: off -> standby, hv on -> trg.  then wait for filament ot heat up.  observe "filament wait time"

typically the station is in standby.

then press high voltage.

rf system tries to ramp up the voltage, charge voltage rb has to match sp.

then once stable (rb charge is match sp charge), press trigger mode.

then when trigger is on, close the feedforward,  in main bc2 overview.

go to amp say 10.

blue line must not exceed 8e-8 in the pressure plot of bc2 tds operation overview panel.

now have stable signal in Vs and SP amplitude plots?

then raise the amplitude up to 20 and check it's all good.

if there was some interruption or long time without using it then maybe need to check ramping is fine.

still in bc2 tds operation overview:

fsm has to be on.  ignore RF on/off.  just turn it on.  actually it does very little.  will say "TDs is off beam" for example.



on/off beam

beam dynamics -> expert -> xfel_b2_diag_bunches.xml. ( diag bunches bc2)

kicker_number = bunch_number = 120
turn tds on.
number of pulses.

start diagnostic bunches

kax switches on kicker system, so don't use it.  or just kicker number to 0.

bunch number to 1, this is always the first stable bunch!.  so this is good.


- if there's a trip for example:
the feedforward will be on.  will be on the maximum amplitude
so:
1. reduce amplitude
2. switch off feed forward.
3. then turn to hv and so on, otherwise it will trip.

click on standby before pressing reset. maybe also.  don't want to come back at fullpower.

if the charge voltage doesn't stabilise after pressing hv on: go to
smaller setpoint, maybe 800,

press gv on.  then go up in steps of t10 whilst in HV mode.

i1:  feedforward shoudl be on, fsm  should be on.

* Matching

I don't need to manually put the thing in myself when using Matthias's tool

Magnet Energizer needs calling, MAG_I1D
Energy manager needs calling
