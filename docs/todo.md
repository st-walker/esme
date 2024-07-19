# HAS TO BE WORKING BY JULY:

	- The slice energy spread measurement
	- Calibration
	- TDS data taking (do not need online analysis...)

# Handle when not getting screen metadata?  e.g. orc.55.i1 when screen is off doens't work whole gui crashes.

my tool:                     bolko:
~600mum/ps @ OTRC.58.I1      450
~200mum/ps @ OTRC.64.I1D     260


default kicker on/off state should depend on whether or not screen is inserted.
add calibration file location to output of main gui
turn adaptive FF back on automatically.
ibfb should not be warning if the screen is on axis.
make aspect ratio toggleable
clear the image when data taking stops/fails/whatever.
draw ROI.
save roi info.
save screen position.
SBML needs to be B2 aware.
everything minus tds should work for B2.
we need correct r12 calculationf for b2.

What if screen is switched off by someone/something else?
Fix standard increments (+0.1 for voltage for example, +1 for amplitude..)

open blms button
blms to group box
dump switch group box.
optics using design kick server.
if on axis then don't use kickers
if off axis then do use kickers
if screen position is moved then one time check and set whether ot use kickers.
gaussian fit for energy axis.
image analysis
get automatic time calibraiton calculation correct and working.
make calibrator and current profiler use the parent widget/thread,.
two point analysis.
current profiler.
make logging work.
fix colour map
an icon
add to jddd.
dispersion measurement.
turn off current/time axis if sbm isn't firing..
warn/forbid bad bunch numbers and bad region numbers




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
