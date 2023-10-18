

## TDS Control and Calibration

1. [ ] Save Points (centres of mass) of TDS calibration phase scan.
2. [ ] Algorithm for moving from point to point.
3. [ ] Algorithm for shifting to edges.
4. [ ] Use the special bunch mid layer to do this stuff.
5. [ ] Let calibrator pause between setpoints.

## HIREIS high resolution energy spread measurement

### High priority:

1. [ ] Dynamically load the TDS calibration.
2. [ ] Get naive implementation based on assumed emittance and screen resolution
3. [ ] Add dispersion measurement.
4. [x] make use of special bunch midlayer optional (but the default)
5. [x] add bunch length to the output
6. [x] Handle it gracefully on valueerror/domain error when input is shit.
7. [x] dso it with the special bunch midlayer so that we don't ahve to
   worry about shit going south, masks work etc. Because THE TDS IS
8. [x] change filenames to local time!  to match logbook...
9. [x] Save background images too beforehand please.
10. [x] Add background data to process.
11. [x] Add beta scan to gui.
12. [x] Button to interrupt/stop.
13. [x] get the results box working.

    
### Middle Priority

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



### Low Priority

1. [ ] "Repeat TDS Scan" button (no need always to do e.g. dispersion scan or whatever...)
2. [ ] get beta/dispersion selection correct.
3. [ ] Full suite of data plots.
4. [x] make sure the config panel works.
5. [x] Add label to dispersion setting label.

