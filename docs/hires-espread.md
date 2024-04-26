
# Checklist for Data Taking

- [ ] Dump switch to set magnets
- [ ] Main Timing to set bunch pattern to I1D
- [ ] RF Sum Voltage Control, set all chirps to 0.0
- [ ] Set Scalar mode
- [ ] Set AH1 to 0.0, ramping A1 in tandem.
- [ ] Turn off AH1
- [ ] Turn off the laser heater.
- [ ] Adjust the TDS phase until it's on crest.
- [ ] Make sure the TDS is on
- [ ] Adjust A1 phase until the beam is symmetrical.

## Dispersion Measurement

Tools -> Measurement Scan Tool -> scanTool

Acutrator: drag voltage, should be scalar mode
[121:0.5:123]
set-read delay: 1 second

open fit studio

![image](../images/dispersion-measurement-doocs-example.png)

multiply by energy to get dispersion.

