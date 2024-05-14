import logging
import time

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal

from esme.gui.ui.special_bunch_panel import Ui_special_bunch_panel
from esme.control.pattern import get_beam_regions, get_bunch_pattern
from .common import get_machine_manager_factory
from esme.core import DiagnosticRegion

LOG = logging.getLogger(__name__)

# SpecialBunchControl doesn't know much/anything about screens, only kickers.


class SpecialBunchMidLayerPanel(QtWidgets.QWidget):
    # This can be quite slow because it's very unlikely that the SBM will change much
    # as a result of something from outside of this GUI.
    MAIN_TIMER_PERIOD_MS = 2500
    def __init__(self, parent=None):
        super().__init__(parent=parent)

        self.i1dbmanager = get_machine_manager_factory().make_diagnostic_bunches_manager(DiagnosticRegion.I1)
        self.b2dbmanager = get_machine_manager_factory().make_diagnostic_bunches_manager(DiagnosticRegion.B2)
        self.dbunch_manager = self.i1dbmanager # Set sbm choice to be for I1 diagnostics

        self.ui = Ui_special_bunch_panel()
        self.ui.setupUi(self)

        self.ifbb_warning_dialogue = IBFBWarningDialogue(self.dbunch_manager.sbunches, parent=self)
        self.ifbb_warning_dialogue.fire_signal.connect(self.unsafe_diagnostic_bunch_start)
        self.ifbb_warning_dialogue.disable_ibfb_aff_signal.connect()

        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(self.MAIN_TIMER_PERIOD_MS)
        self.update_ui()

    def update_ui(self) -> None:
        """Read values from the DOOCs server and update the daughter widget states here."""
        # Add 1 because the SBM is zero-counting for the beam regions but we always speak
        # of the first beam region being the first, not zeroth.
        self.ui.beamregion_spinbox.setValue(self.dbunch_manager.sbunches.get_beam_region() + 1)
        self.ui.bunch_spinbox.setValue(self.dbunch_manager.sbunches.get_bunch_number())
        self.ui.npulses_spinbox.setValue(self.dbunch_manager.sbunches.get_npulses())
        self.ui.ibfb_checkbox.setCheckState(self.get_ibfb_checkstate())
        self.ui.use_tds_checkbox.setChecked(self.dbunch_manager.sbunches.get_use_tds())
        self.ui.use_fast_kickers_checkbox.setChecked(self.check_fast_kickers_state())
        # self.check_start_stop()

    def get_ibfb_checkstate(self) -> Qt.CheckState:
        xon = self.dbunch_manager.sbunches.ibfb_x_lff_is_on()
        yon = self.dbunch_manager.sbunches.ibfb_y_lff_is_on()
        if xon and yon:
            return Qt.Checked
        if xon ^ yon:
            return Qt.PartiallyChecked
        return Qt.Unchecked

    def check_fast_kickers_state(self) -> None:
        print(self.dbunch_manager.sbunches.would_use_kickers())
        if self.dbunch_manager.sbunches.would_use_kickers():
            return True
        return False

    def set_bunch_control_enabled(self, enabled: bool) -> None:
        """Enable or disable UI elements based for modification,
        typically depending on whether or not the kicker/tds system is
        currently firing."""
        self.ui.beamregion_spinbox.setEnabled(enabled)
        self.ui.bunch_spinbox.setEnabled(enabled)
        self.ui.go_to_last_bunch_in_br_pushbutton.setEnabled(enabled)
        self.ui.go_to_last_laserpulse_pushbutton.setEnabled(enabled)
        self.ui.use_fast_kickers_checkbox.setEnabled(enabled)
        self.ui.use_tds_checkbox.setEnabled(enabled)
        self.ui.npulses_spinbox.setEnabled(enabled)

    def connect_buttons(self) -> None:
        """Just called during __init__ for where UI button callbacks are set."""
        self.ui.use_fast_kickers_checkbox.stateChanged.connect(self.toggle_kickers)
        self.ui.use_tds_checkbox.stateChanged.connect(self.dbunch_manager.sbunches.set_use_tds)
        self.ui.beamregion_spinbox.valueChanged.connect(lambda n: self.dbunch_manager.sbunches.set_beam_region(n - 1))
        self.ui.bunch_spinbox.valueChanged.connect(self.dbunch_manager.sbunches.set_bunch_number)
        self.ui.npulses_spinbox.valueChanged.connect(self.dbunch_manager.sbunches.set_npulses)
        self.ui.go_to_last_laserpulse_pushbutton.clicked.connect(self.goto_last_bunch_in_machine)
        self.ui.go_to_last_bunch_in_br_pushbutton.clicked.connect(self.goto_last_bunch_in_beam_region)
        self.ui.ibfb_checkbox.stateChanged.connect(self.set_ibfb_state)

        self.ifbb_warning_dialogue.fire_signal.connect(self.dbunch_manager.sbunches.start_diagnostic_bunch)
        self.ifbb_warning_dialogue.disable_ibfb_aff_signal.connect(lambda: self.dbunch_manager.sbunches.set_ibfb_state(on=False))

        # Connections happen in this method as button is a toggle:
        self.check_start_stop()

    def set_ibfb_state(self, state: Qt.CheckState) -> None:
        """Intended as a function to be connected to the IBFB checkbox.
        Writes the desired state to the DOOCS server."""
        self.dbunch_manager.sbunches.set_ibfb_lff(on=bool(state))

    def check_start_stop(self) -> None:
        """Check whether or not the diagnostic bunch is currently firing."""

        if self.dbunch_manager.sbunches.is_diag_bunch_firing(): # If Firing
            self.ui.start_button.setText("Stop Diag. Bunch")
            try: # Disconnect any connections, raises if there are none so we catch and pass
                self.ui.start_button.clicked.disconnect()
            except TypeError:
                pass
            self.ui.start_button.clicked.connect(self.dbunch_manager.sbunches.stop_diagnostic_bunch)
            self.set_bunch_control_enabled(False)
        else:
            self.ui.start_button.setText("Start Diag. Bunch")
            try:
                self.ui.start_button.clicked.disconnect()
            except TypeError:
                pass
            self.ui.start_button.clicked.connect(self.safe_diagnostic_bunch_start)
            self.set_bunch_control_enabled(True)

    def safe_diagnostic_bunch_start(self) -> None:
        """The diagnostic bunch should not be fired whilst the IBFB LFF is activated, otherwise the IBFB will
        try and counter the impact """
        if self.dbunch_manager.sbunches.is_either_ibfb_on():
            self.ifbb_warning_dialogue.show()
        self.dbunch_manager.sbunches.start_diagnostic_bunch()

    def unsafe_diagnostic_bunch_start(self) -> None:
        self.dbunch_manager.sbunches.start_diagnostic_bunch()

    def toggle_kickers(self, state: Qt.CheckState) -> None:
        # Qt.Unchecked == 0
        # Qt.PartiallyChecked: Assumed to never be.
        # Qt.Checked == 2
        assert state != Qt.PartiallyChecked
        if state:
            self.dbunch_manager.sbunches.do_use_kickers()
        else:
            self.dbunch_manager.sbunches.dont_use_kickers()

    def set_kickers_for_screen(self, screen_name: str) -> None:
        self.dbunch_manager.set_kickers_for_screen(screen_name)

    def set_use_fast_kickers(self, kicker_name: str) -> None:
        """Set whether the fast kicker(s) should fire when the
        diagnostic bunch is fired.  To do this"""
        # Get the current kicker setpoint, for some reason...
        ksp = self.kicker_setpoint
        if self.ui.use_fast_kickers_checkbox.isChecked() and self.kicker_setpoint is not None:
            if ksp is None:
                return
            kicker_name = ksp[0].name
            # LOG.info(f"Enabling fast kickers for {self.screen_name}: kicker: {kicker_name}")
            # Just use first kicker name and assume that all are then
            # set (i.e they have the same kicker numbers---they should
            # be configured as such on the doocs server...).
            self.dbunch_manager.sbunches.set_kicker_name(kicker_name)
            # Power on kickers if they are to be used.
            if not self.dbunch_manager.sbunches.would_use_kickers():
                self.dbunch_manager.sbunches.power_on_kickers()
        else:
            # Set kickers not to be used in the SBM.
            self.dbunch_manager.sbunches.dont_use_kickers()

    def goto_last_bunch_in_machine(self) -> None:
        beam_regions = get_beam_regions(get_bunch_pattern())
        last_beam_region = beam_regions[-1]
        nbunches = last_beam_region.nbunches()
        beam_region_number = last_beam_region.idn
        diagnostic_bunch_number = nbunches + 1
        # assert last_beam_region > 0
        LOG.info(f"Found last bunch in machine: BR = {beam_region_number}, last normal bunch no. = {nbunches}, diagnostic bunch no. = {diagnostic_bunch_number}")
        self.dbunch_manager.sbunches.set_beam_region(beam_region_number - 1)
        self.dbunch_manager.sbunches.set_bunch_number(diagnostic_bunch_number)
        self.update_ui()

    def goto_last_bunch_in_beam_region(self) -> None:
        beam_regions = get_beam_regions(get_bunch_pattern())
        # This is zero counting!! beam region 1 is 0 when read from sbunch midlayer!
        selected_beam_region = self.dbunch_manager.sbunches.get_beam_region()
        assert selected_beam_region >= 0
        try:
            br = beam_regions[selected_beam_region]
        except IndexError:
            LOG.info(f"User tried to select last bunch of nonexistent beam region: {selected_beam_region}.")
            box = QMessageBox(self) #, "Invalid Beam Region",
            box.setText(f"Beam Region {selected_beam_region+1} does not exist.")
            box.exec()
            return
        else:
            self.dbunch_manager.sbunches.set_bunch_number(br.nbunches())
        self.update_ui()

    # def update_panel_start_stop_state(self):
    #     is_diag_bunch_firing = self.dbunch_manager.sbunches.is_diag_bunch_firing()
    #     # print(f"{is_diag_bunch_firing=}")
    #     # if is_diag_bunch_firing:
    #     #     self.set_bunch_control_enabled(False)
    #     #     self.ui.start_button.setEnabled(False)
    #     #     self.ui.stop_button.setEnabled(True)
    #     # else:
    #     #     self.set_bunch_control_enabled(True)
    #     #     self.ui.start_button.setEnabled(True)
    #     #     self.ui.stop_button.setEnabled(False)

    #     if is_diag_bunch_firing:
    #         self.set_bunch_control_enabled(False)
    #         self.ui.start_button.setEnabled(False)
    #         self.ui.stop_button.setEnabled(True)
    #     else:
    #         self.set_bunch_control_enabled(True)
    #         self.ui.start_button.setEnabled(True)
    #         self.ui.stop_button.setEnabled(False)



class IBFBWarningDialogue(QMessageBox):
    SHORT_TEXT = """IBFB Adaptive FF is still on.  Disable and start firing diagnostic bunches?"""
    LONG_TEXT = ("The IBFB Adaptive Feed Forward has"
                 " not been disabled but you are trying to fire a diagnostic bunch."
                 " It is generally advisable to disable IBFB Adaptive FF before firing a diagnostic"
                 " bunch, otherwise the IBFB will try and counteract the TDS and"
                 " fast kicker kicks, which will cause problems all along the bunch train"
                 " and result in an otherwise parasitic measurement becoming invasive.")
    fire_signal = pyqtSignal()
    disable_ibfb_aff_signal = pyqtSignal()

    def __init__(self, sbinterface, parent=None):
        super().__init__(parent=parent)

        self.sbinterfaces = sbinterface

        self.setIcon(QMessageBox.Warning)
        self.setWindowTitle('Warning')
        self.setText(self.SHORT_TEXT)
        self.setDetailedText(self.LONG_TEXT)

        # Custom buttons
        self.disable_ibfb_lff_button = self.addButton("OK", QMessageBox.ActionRole)
        self.ignore_and_go_button = self.addButton("Ignore", QMessageBox.ActionRole)
        self.cancel_button = self.addButton("Cancel", QMessageBox.RejectRole)

        # Connect buttons to functions
        self.buttonClicked.connect(self.on_button_clicked)

    def on_button_clicked(self, button):
        if button == self.disable_ibfb_lff_button:
            self.sbinterfaces.set_ibfb_state(on=False)
            time.sleep(0.1)
            self.sbinterfaces.start_diagnostic_bunch()
        elif button == self.ignore_and_go_button:
            self.sbinterfaces.start_diagnostic_bunch()
        elif button == self.cancel_button:
            self.hide()
