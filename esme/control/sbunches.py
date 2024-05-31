"""Module for controlling the Special Bunch Midlayer (SBM) for diagnostic bunches

It is responsible for firing diagnostic bunches and optionally firing
kickers and the TDS in tandem with these diagnostic bunches.

This module is designed to be an easy way to set the SBM for the desired use:

1. Select a beam region and bunch number to be kicked, streaked, both or neither.
2. Select a kicker or kickers to fire (or not, if e.g. for an on-axis screen).
3. Select whether to use the TDS in that area (or not if you don't want the beam streaked).

Screens
-------

The SBM is generally screen-agnostic and doesn't know much about
screens (although there is some duplication here---see
`SpecialBunchControl.is_screen_ok`).  It assumes that the kicker
voltages and timings are configured correctly if you have a specific
screen in mind.

If the fast kickers ARE used then the screen should be OFF-axis.  
If the fast kickers ARE NOT used then the screen should be ON-axis.

Control list
------------

Most addresses are somewhat self explanatory, but the control list should be explained.
In principle you do not need to know about this as the class below takes care of it,
however for posterity I include it here to save you the effort I myself made.

The addresses for I1 and B2, respectively, are:

XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/CONTROL
XFEL.SDIAG/SPECIAL_BUNCHES.ML/B2/CONTROL

Each is a 4-tuple of integers.  Each entry of this tuple has a different meaning. In order:

0. The bunch number of the diagnostic bunch of the given subtrain.
   The kickers and TDS will (if desired) be fired for this bunch.
   It can be at most n+1, where n is the number of bunches in the
   selected subtrain.  If equal to n+1, then a bunch is appended
   to the bunch train, if greater than n+1, then when the diagnostic
   bunch is activated, nothing will happen.
1. Whether to use the TDS for the diagnostic bunch.
   0 = don't use, 1 = use.
2. The ID of the kicker number we will fire for the diagnostic bunch.
   At the time of writing, an integer between 1 and 4.
3. Doesn't do anything.


Kicker Activation
-----------------

One tricky detail is how we choose whether or not to use the kickers when using the SBM.
The JDDD panels (Beam Dynamics -> Expert Panels -> Diag Bunches Inj / Diach Bunches BC2)
have the checkbox "KAX on/off" (for I1: XFEL.SDIAG/SPECIAL_BUNCHES.ML/54.I1/KICKER.ON)
for determining whether or not the kickers should be _powered_.  But that does not mean
the kickers will necessarily be used, the "kicker number" (also found on the above JDDD panels)
of the control 4-tuple (e.g. for I1, address = XFEL.SDIAG/SPECIAL_BUNCHES.ML/I1/CONTROL)
must also be set to a kicker number for which there exists a corresponding kicker with that number
assigned to it in DOOCS.  Using I1 as an example, the kicker numbers of the two kickers in I1
can be read using the pydoocs read (or by using
SpecialBunchesControl.get_kicker_name_to_kicker_number_map below):

```
pydoocs.read("XFEL.SDIAG/SPECIAL_BUNCHES.ML/K*.I1/KICKER_NUMBER")
```

where (at the time of writing) the kicker numbers of KAX.54.I1 and KAX.56.I1
can be seen to be 1 and 3, respectively.  There are no other diagnostic kickers in I1.

Therefore if we set the kicker number in the control address
(or by using SpecialBunchesControl.set_kicker_name below) to 1, then KAX.54.I1 will be fired
with the SBM.  If 3, then KAX.56.I1, but if 2 then no kicker will be fired.  In short: this means
that to disable the kickers, the kickers must be powered off OR the kicker number
of the I1 control list in this case must be set to any number except 1 and 3.  Conversely, if the kickers
are on, then the kicker number must be set to 1 or 3.  

In general, we prefer to leave the kickers on as powering them on/off takes times, so controlling
kicker activation is done by changing this kicker number to and from a small set of integers (at the time
of writing: 1 or 3 for I1 and 1, 2, 3 or 4 in B2).

I1 or B2
--------

The SBM is specific to the two diagnostic regions as well, either in
I1 or B2.  It is meaningless to mix these two, e.g. firing the TDS in
the injector alongside kickers in B2.  Whether or not a given class instance is 
responsible for I1 or B2 is defined in the __init__.

"""


from typing import Optional
from collections import defaultdict
import logging

from .dint import DOOCSInterface

from esme.core import DiagnosticRegion

LOG = logging.getLogger(__name__)


class SpecialBunchesControl:
    STATUS_IS_OK_VALUE = 0
    DIAG_BUNCH_FIRE_VALUE = 1
    DIAG_BUNCH_STOP_VALUE = 0
    LFF_X_ADDRESS = "XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/ADAPTIVE_FF_EN_X"
    LFF_Y_ADDRESS = "XFEL.DIAG/DAMC2IBFB/DI1914TL.0_CTRL/ADAPTIVE_FF_EN_Y"

    def __init__(self, location: DiagnosticRegion,
                 kicker_number: int | None = None,
                 di: DOOCSInterface | None = None) -> None:
    
        self.location = location or DiagnosticRegion.I1
        self.di = di or DOOCSInterface()
        self._defined_kicker_numbers = self._kicker_number_area_map()
        # We try to overcome a "deficiency" in the SBM whereby the kicker selection and whether
        # or not we will use the kickers are done with the same interface (a single integer).
        # Ideally we'd have a selected kicker AND a toggle.  This would be a nicer interface as
        # So we store some selected kicker here and then.
        self._kicker_number = self._first_valid_kicker_number()

    def _first_valid_kicker_number(self) -> int:
        return min(self._kicker_number_area_map()[self.location])
    
    @property
    def kicker_number(self) -> int:
        return self._kicker_number
    
    @kicker_number.setter
    def kicker_number(self, n: int | None) -> None:
        # What is this bit of code doing?  Why can None be a possible Kicker number?
        if n is not None and n not in self._kicker_number_area_map()[self.location]:
            raise ValueError(f"Selected kicker number {n} does not correspond to a real kicker.")
        self._kicker_number = n

    def set_beam_region(self, br: int) -> None:
        """Set the beam region of the special bunch that would be fired.  Note that the number here is zero counting!"""
        ch = self.beamregion_address()
        LOG.info(f"Setting beam region (zero counting here): {ch} = {br}")
        self.di.set_value(ch, br)

    def get_beam_region(self) -> None:
        """Note this is zero counting!"""
        ch = self.beamregion_address()
        return self.di.get_value(ch)

    def set_npulses(self, npulses: int) -> None:
        """Set the number of diagnostic bunches that would be fired."""
        ch = self.npulses_address()
        LOG.info(f"Setting npulses: {ch} = {npulses}")
        self.di.set_value(ch, npulses)

    def get_npulses(self) -> int:
        """Get the number of diagnostic bunches that would be fired."""
        ch = self.npulses_address()
        return self.di.get_value(ch)

    def get_kicker_name_to_kicker_number_map(self) -> dict[str, int]:
        """Kickers to be fired are selected using kicker numbers in the SBM, not kicker names.
        Each kicker in the SBM has a number.  If there are two kickers with the same number,
        and that number is selected for the SBM, then that means both kickers will be fired
        for the given diagnostic bunch.  This is not so relevant in I1, where no more than
        one kicker is ever used at a time to kick onto a screen, but in B2 two kickers at a
        time are used in tandem to reach a given screen.  So pairs of kickers with the same
        kicker number can be selected to fire when the diagnostic bunch is fired."""
        # Get all the kicker numbers for the given region.
        knumbers = self.di.get_value("XFEL.SDIAG/SPECIAL_BUNCHES.ML/K*/KICKER_NUMBER")
        rdict = {}
        # Map the kicker names to kicker numbers
        for kicker_number, *_, kicker_name in knumbers:
            rdict[kicker_name] = kicker_number
        LOG.debug(f"Built kicker name to kicker index map: {rdict}")
        return rdict
    
    def _kicker_number_area_map(self) -> dict[str, set[int]]:
        """Build a map of diagnostic regions to sets of kicker numbers.  The use of this function is to
        see which kicker numbers are defined for each diagnostic region.  This is important as we use whether
        or not such kicker numbers are defined as equivalent to whether or not any kicker should be fired."""
        kicker_numbers_per_area = defaultdict(set)
        for kicker_name, kicker_number in self.get_kicker_name_to_kicker_number_map().items():
            # XXX: We assume the last 2 characters are either I1, B1 or B2 and tell us which area the kicker is in
            area = kicker_name[-2:]
            kicker_numbers_per_area[area].add(kicker_number)
        return kicker_numbers_per_area
    
    def get_control_list(self) -> list[int]:
        """Return the control list for the set DiagnosticRegion.  
        The control list is a 4-tuple:
        
        0. The bunch number of the diagnostic bunch counting from start of the current beam region
        1. Whether to use the tds.
        2. The kicker number of the kicker to be fired, if set to a number for which there is no kicker
           with that number, then no kicker is fired.
        3. nothing.
        """
        return self.di.get_value(self.control_address())

    def set_bunch_number(self, bn: int) -> None:
        """Set the bunch number of the diagnostic bunch counting from the beam region."""
        clist = self.get_control_list()
        clist[0] = int(bn)
        self.di.set_value(self.control_address(), clist)

    def get_bunch_number(self) -> int:
        """Get the bunch number of the diagnostic bunch for the chosen beam region (see: get_beam_region)."""
        clist = self.get_control_list()
        return int(clist[0])

    def set_use_tds(self, use_tds: bool) -> None:
        """Set whether to use the TDS when the diagnostic bunch is fired."""
        clist = self.get_control_list()
        clist[1] = int(bool(use_tds))
        self.di.set_value(self.control_address(), clist)

    def get_use_tds(self) -> bool:
        """Read whether the SBM is configured to fire the TDS for diagnostic bunches."""
        clist = self.get_control_list()
        return bool(clist[1])

    def set_kicker_name(self, kicker_name: str) -> None:
        """Select a kicker given by name for use in the special bunch midlayer.
           This sets the relevant entry in the control address (a 4-tuple of integers) so that when a diagnostic bunch is fired, 
           IF the kickers are ALSO enabled (using method power_on_kickers).

           If there are two kickers we want to set (i.e. two with the same kicker number), then we only need to set one of them, as they
           share the same kicker number, so this only needs to be called once in that case, although calling it again will not cause 
           any problems.

        See Also
        --------
        would_use_kickers
        dont_use_kickers
        do_use_kickers

        """
        clist = self.get_control_list()
        kmap = self.get_kicker_name_to_kicker_number_map()
        kicker_number = kmap[kicker_name]
        clist[2] = kicker_number
        LOG.info(f"Enabling fast kicker by setting kicker number in {self.control_address()}, {kicker_name=}, {kicker_number=}")
        self.di.set_value(self.control_address(), clist)

    # def set_kicker_number_by_name(self, kicker_name: str) -> None:
    #     if self.kicker_number is None:
    #         raise ValueError("No kicker number to write to CONTROL.")
    #     clist = self.get_control_list()
    #     kmap = self.get_kicker_name_to_kicker_number_map()
    #     kicker_number = kmap[kicker_name]
    #     clist[2] = kicker_number
    #     LOG.info(f"Enabling fast kicker by setting kicker number in {self.control_address()}, {kicker_name=}, {kicker_number=}")
    #     self.di.set_value(self.control_address(), clist)
    #     pass

    def _write_kicker_number_to_sbm(self, kn: int) -> None:
        clist = self.get_control_list()        
        clist[2] = kn
        self.di.set_value(self.control_address(), clist)

    def do_use_kickers(self) -> None:
        """Set the kickers to use the instance attribute `kicker_number`."""
        kn = self.kicker_number
        LOG.info("Enabling kicker(s) in SBM by writing kicker number %d to SBM", kn)
        self._write_kicker_number_to_sbm(kn)

    def dont_use_kickers(self) -> None:
        """This does not power the kickers down, just sets them not to be used when a diagnostic bunch is fired.
        See Also
        --------
        would_use_kickers
        do_use_kickers

        """
        kn = self._get_unused_kicker_number()
        LOG.info("Disabling kicker(s) in SBM by writing unused kicker number %d to SBM", kn)
        self._write_kicker_number_to_sbm(kn)

    def would_use_kickers(self):
        """Whether or not the kicker(s) will fire when the diagnostic bunch is started depends on two things:

        1. The kicker(s) must be powered on.
        2. The kicker number in the control 4-tuple must be set to a number that "exists".  
           That is, there must be at least one kicker in that diagnostic region with its kicker number set to that value.
        
        e.g. if the kickers in I1 have the numbers 1 and 3, but the kicker number in the control list is set to 2, 
        then no kickers will fire, regardless of whether or not they are powered on.

        See Also
        --------
        set_kicker_name (to set a kicker to fire)
        dont_use_kickers (to not use kickers)

        """
        powered = self.are_kickers_powered_up()
        active_kicker_number = self._get_active_kicker_number()
        active_kicker_number_corresponds_to_real_kicker =  active_kicker_number in self._defined_kicker_numbers[self.location.name]
        LOG.debug(f"Would not use kickers.  Reason: {powered=} and {active_kicker_number_corresponds_to_real_kicker=} ({active_kicker_number=})")
        return powered and active_kicker_number_corresponds_to_real_kicker
   
    def _get_active_kicker_number(self) -> int:
        """Which kicker number is set in the SBM to (try to) be used if the diagnostic bunch fires?"""
        clist = self.get_control_list()
        return clist[2]
    
    def _get_unused_kicker_number(self) -> int:
        """Return an unused kicker number.  Used for disabling the kickers."""
        return max(self._defined_kicker_numbers[self.location.name]) + 1

    def are_kickers_powered_up(self) -> bool:
        """Check whether the kicker is powered on or not for the diagnostic region.

        See Also
        --------
        power_up_kickers
        power_down_kickers 
        """
        value = False
        # Get the relevant address for whether the kickers are on by wildcard
        for readout in self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/*{self.location.name}/KICKER.ON"):
            value, *_, loc = readout
            _, location = loc.split(".")
            # we have found the address corresponding to our region
            if location == self.location:
                break
        # The kickers have to be on AND
        ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{self.location.name}/KICKER.ON"
        LOG.debug(f"Read {value} from {ch}")
        return bool(value)

    def power_up_kickers(self) -> None:
        """Power up the kickers.
        
        See Also
        --------
        power_down_kickers
        are_kickers_powered_up
        """
        self._power_up_down_kickers(on=True)

    def power_down_kickers(self) -> None:
        """Power down the kickers.
        
        See Also
        --------
        power_up_kickers
        are_kickers_powered_up
        """
        self._power_up_down_kickers(on=False)

    def _power_up_down_kickers(self, on: bool = True):
        # We look up the address for powering the kickers in a somewhat lazy way by using a wildcard. 
        # This works for both diagnostic regions and means we don't have to hardcode additional addresses
        # for each region, we just look them up here at run time.
        *_, loc = self.di.get_value(f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/*{self.location.name}/KICKER.ON")[0]
        ch = f"XFEL.SDIAG/SPECIAL_BUNCHES.ML/{loc}/KICKER.ON"
        value = int(bool(on)) # Coerce non-zero/truthy to 1 and zero/falsy to 0.
        self.di.set_value(ch, value)

    def control_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/CONTROL".format(self.location.name)

    def beamregion_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/SUBTRAIN".format(self.location.name)

    def status_address(self, thing: str) -> str:
        """Check the status of the TDS, CAMERA or KICKER ("thing") from the view point of the SBM."""
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/STATUS.{}".format(self.location.name, thing)

    def npulses_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/PULSES.ACTIVE".format(self.location.name)

    def fire_diagnostic_bunch_address(self) -> str:
        return "XFEL.SDIAG/SPECIAL_BUNCHES.ML/{}/START".format(self.location.name)

    def is_tds_ok(self) -> bool:
        """Check whether the SBM is happy with the state of the TDS."""
        value = self.di.get_value(self.status_address("TDS"))
        return value == self.STATUS_IS_OK_VALUE

    def is_screen_ok(self) -> bool:
        """Check whether the SBM is happy with the state of the screen."""
        value = self.di.get_value(self.status_address("CAMERA"))
        return value == self.STATUS_IS_OK_VALUE

    def is_kicker_ok(self) -> bool:
        """Check whether the SBM is happy with the state of the kicker(s)."""
        value = self.di.get_value(self.status_address("KICKER"))
        return value == self.STATUS_IS_OK_VALUE

    def is_diag_bunch_firing(self) -> bool:
        """Check whether the diagnostic bunch is currently firing"""
        ch = self.fire_diagnostic_bunch_address()
        return self.di.get_value(ch) == self.DIAG_BUNCH_FIRE_VALUE

    def start_diagnostic_bunch(self) -> None:
        """Start firing diagnostic bunches"""
        ch = self.fire_diagnostic_bunch_address()
        LOG.info(f"Starting diagnostic bunches: {ch} = {self.DIAG_BUNCH_FIRE_VALUE}")
        self.di.set_value(ch, self.DIAG_BUNCH_FIRE_VALUE)

    def stop_diagnostic_bunch(self) -> None:
        """Stop firing diagnostic bunches"""
        ch = self.fire_diagnostic_bunch_address()
        LOG.info(f"Stopping diagnostic bunches: {ch} = {self.DIAG_BUNCH_STOP_VALUE}")
        self.di.set_value(ch, self.DIAG_BUNCH_STOP_VALUE)

    def ibfb_x_lff_is_on(self) -> bool:
        """Check whether learning feed forward (LFF) is active for IBFB in x-plane"""
        return bool(self.di.get_value(self.LFF_X_ADDRESS))

    def ibfb_y_lff_is_on(self) -> bool:
        """Check whether learning feed forward (LFF) is active for IBFB in y-plane"""
        return bool(self.di.get_value(self.LFF_Y_ADDRESS))

    def is_either_ibfb_on(self) -> bool:
        """Check whether IBFB is on in either plane"""
        return self.ibfb_x_lff_is_on() or self.ibfb_y_lff_is_on()

    def set_ibfb_lff(self, *, on: bool) -> None:
        """Set the IBFB LFF on or off."""
        state = int(bool(on))
        self.di.set_value(self.LFF_X_ADDRESS, state)
        self.di.set_value(self.LFF_Y_ADDRESS, state)
