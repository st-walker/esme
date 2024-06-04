questions for bolko:



what are in bolko's .mat files from his calib?

what can image analysis server actually do?
can we make the image analysis server easier to use?


I didn't fix the problem with the kickers being set properly...
i need to do it via the machine so that the screen is set,
i got rid of this at some point.

save roi info but also raw data?
set roi manually?wh

# LPS Main GUI

## Minimal feature set todolist (Highest Priority)

1. [ ] Use Bolko Calibrations.
2. [x] Use new python12 pydoocs without GIL.
3. [x] Be able to take data.


## High Priortiy

1. [ ] Find phase.
2. [ ] Reconstruct current profile using both crossings.
3. [x] Subtract background buton
3. [x] Get transverse projections working.
4. [x] Get time projection working.
5. [x] Get energy projection working.
6. [x] Subtract background button.
7. [x] ensure SBP behaves itself when stopping
8. [x] IBFP Check




## Middle Priority

1. [x] Get plain data taking working.
2. [ ] bunch pattern to and from I1D or B2D.
3. [ ] Design kick server, to and from 200m optics, to and from I1D.
4. [ ] Automatically Calibrate TDS.
5. [X] Get negative axis working.
6. [x] add author string to print to logbook.
7. [ ] figure out levels stuff so images look good.
8. [ ] add external panels to top menu, camera status, pattern builder, tds monitor, special bunch diagnostic?
9. [x] play / pause / set rate data acquisition
10. [ ] ROI.

## Low Priority

1. [x] Get indiciators working.
2. [ ] Dispersion measurement.
3. [ ] Use Kickers to steer onto screen automatically (i.e. before TDS is ever switched on).
4. [x] Handle screens being on/off properly (including turning off afterwards if we turned them on)
5. [ ] add some sort of help, e.g., remember to update the timings in the config file...
6. [x] make initial size 50% bigger
7. [ ] get rid of logging tab somehow.
8. [ ] reduce spacing between control boxes a little bit.
9. [x] Change JDDD... to Open JDDD panel


# High Resolution Slice Energy Spread

## High priority:

- [x] Dynamically load the TDS calibration.
- [ ] Get naive implementation based on assumed emittance and screen resolution
- [ ] Add dispersion measurement.
- [x] make use of special bunch midlayer optional (but the default)
- [x] add bunch length to the output
- [x] Handle it gracefully on valueerror/domain error when input is shit.
- [x] Do it with the special bunch midlayer so that we don't have to deal with BLMs
- [x] change filenames to local time!  to match logbook...
- [x] Save background images too beforehand please.
10. [x] Add background data to process.
11. [x] Add beta scan to gui.
12. [x] Button to interrupt/stop.
13. [x] get the results box working.
2. [ ] Add timestmap to output.
3. [ ] Full slice Measurements
4. [ ] Add editable dispersion based on measurement (not hardcoded to linear optics).
5. [ ] Add Laser Heater Status
6. [ ] Fix bug with TDS tickbox going out of date in SBM panel.
7. [ ] Refuse the loading of projected match .mat file.
8. [ ] Add Beam on/off buttons, add TDS on/off buttons.
20. [ ] Open output directory button
21. [ ] LH indicator
22. [ ] also write which version of esme is being used and where?e
23. [ ] copy over scan.yaml as well also writing the calibration to the scan.yaml file.


## Middle Priority

1. [ ] Optimise dscan/tscan/beta scan diplicate datapoints...
2. [ ] Show the selected slice.
3. [ ] Add checklist to code.
4. [x] Handle case where measurement is started whilst config is open.
5. [x] copy configs to output as well for completion please.
6. [x] write comment box also to file in output directory.
7. [x] put directory location in text box
8. [x] Use a proper table for box vis.
9. [x] Add cycle ability to quadrupoles.
10. [x] write text box to output
11. [x] Just crash casually if beam goes off unexpectedly.
12. [x] Need to control TDS from the panel
13. [x] Button to open the camera server (CHECK!)
14. [ ] get buttons working for dispersion/beta selection.


## Low Priority

1. [ ] "Repeat TDS Scan" button (no need always to do e.g. dispersion scan or whatever...)
2. [ ] get beta/dispersion selection correct.
3. [ ] Full suite of data plots.
4. [x] make sure the config panel works.
5. [x] Add label to dispersion setting label.


# TDS Calibration

1. [ ] Algorithm for moving from point to point (to ramp).
2. [ ] Use the special bunch mid layer to do this stuff.
3. [ ] Save EVERYTHING for this measurement.
4. [ ] Algorithm for shifting to edges.
5. [ ] write the calibration to file at the end.





# Data Explorer

## High Priority

- [ ] Change the TDS Calibration.
- [ ] Get image analysis actually working.
- [X] Jump to result.
- [ ] Mask images.
- [ ] Show slice energy spread.
- [ ] Add outlier filtering


## Medium Priority

- [ ] R34 plot.
- [ ] Plot TDS voltages
- [ ] Plot streaking parameters
- [ ] plot apparent bunch lengths
- [ ] plot actual bunch lengths
- [ ] plot fits.
- [X] Highlight line when I click on it
- [ ] Change number of slices (slider).
- [ ] Change slice algorithm (position versus max energy based).
- [ ] Mask Setpoints.
- [ ] Also show background images.
- [ ] Close children when parent is closed.

## Low Priority

- [ ] Highlight in image ana breakdown when i click row.
- [ ] Use multiprocessing or multithreading.
- [ ] Save plot, save plot data...
- [ ] should TDS scan be linear or not?
- [ ] Maintain image processing stage across different image viewings.
- [ ] Show in window what the "current" slice energy spread is at each stage.
- [ ] Force cropping when image analysis is on.  otherwise it makes no sense anyway!
