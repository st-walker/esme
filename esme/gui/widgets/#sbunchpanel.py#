# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,
import logging
import textwrap

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import pyqtSignal

from .ui.special_bunch_panel import Ui_special_bunch_panel
from esme.control.pattern import get_beam_regions, get_bunch_pattern
from esme.gui.common import make_default_sbm,  set_machine_by_region
from esme.control.screens import FastKickerSetpoint
from esme.core import DiagnosticRegion

LOG = logging.getLogger(__name__)

# SpecialBunchControl doesn't know much/anything about screens, only kickers.


class SpecialBunchMidLayerPanel(QtWidgets.QWidget):
    def __init__(self, kicker_setpoint=None, parent=None):
        super().__init__(parent=parent)

        self.sbinterface = make_default_sbm(location=DiagnosticRegion.I1)
        self.kicker_setpoint = kicker_setpoint

        self.ui = Ui_special_bunch_panel()
        self.ui.setupUi(self)

        self.ifbb_warning_dialogue = IBFBWarningDialogue(self.sbinterface, parent=self)

        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(500)

    def set_kicker_setpoint(self, ksp: list[FastKickerSetpoint]) -> None:
        self.kicker_setpoint = ksp

    def configure_from_screen_name(self, screen_name):
        region = region_from_screen_name(screen_name)
        set_machine_by_region(self, region)
        self.set_use_fast_kickers()

    def update_ui(self):
        self.ui.beamregion_spinbox.setValue(self.sbinterface.get_beam_region() + 1)
        self.ui.bunch_spinbox.setValue(self.sbinterface.get_bunch_number())
        self.ui.npulses_spinbox.setValue(self.sbinterface.get_npulses())
        self.ui.ibfb_checkbox.setChecked(self.sbinterface.is_either_ibfb_on())
        self.ui.use_tds_checkbox.setChecked(self.sbinterface.get_use_tds())
        self.ui.use_fast_kickers_checkbox.setChecked(self.sbinterface.get_use_kicker())
        self.check_start_stop()

    def set_bunch_control_enabled(self, enabled):
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

    def connect_buttons(self):
        """Just called during __init__ for where UI button callbacks are set"""
        self.ui.use_fast_kickers_checkbox.stateChanged.connect(self.set_use_fast_kickers)
        self.ui.use_tds_checkbox.stateChanged.connect(self.sbinterface.set_use_tds)
        self.ui.beamregion_spinbox.valueChanged.connect(lambda n: self.sbinterface.set_beam_region(n - 1))
        self.ui.bunch_spinbox.valueChanged.connect(self.sbinterface.set_bunch_number)
        self.ui.npulses_spinbox.valueChanged.connect(self.sbinterface.set_npulses)
        self.ui.go_to_last_laserpulse_pushbutton.clicked.connect(self.goto_last_bunch_in_machine)
        self.ui.go_to_last_bunch_in_br_pushbutton.clicked.connect(self.goto_last_bunch_in_br)
        self.ui.ibfb_checkbox.stateChanged.connect(self.set_ibfb_state)

        self.ifbb_warning_dialogue.fire_signal.connect(self.sbinterface.start_diagnostic_bunch)
        self.ifbb_warning_dialogue.disable_ibfb_aff_signal.connect(lambda: self.sbinterface.set_ibfb_state(on=False))

        # Connections happen in this method as button is a toggle:
        self.check_start_stop()

    def set_ibfb_state(self, state: Qt.CheckState) -> None:
        self.sbinterface.set_ibfb_lff(on=bool(state))

    def check_start_stop(self) -> None:
        if self.sbinterface.is_diag_bunch_firing(): # If Firing
            self.ui.start_button.setText("Stop Diag. Bunch")
            self.ui.start_button.clicked.connect(self.sbinterface.stop_diagnostic_bunch)
            self.ui.start_button.clicked.disconnect()
            self.set_bunch_control_enabled(False)
        else:
            self.ui.start_button.setText("Start Diag. Bunch")
            self.ui.start_button.clicked.connect(self.safe_diagnostic_bunch_start)
            self.ui.start_button.clicked.disconnect()
            self.set_bunch_control_enabled(True)

    def safe_diagnostic_bunch_start(self):
        if self.sbunches.is_either_ibfb_on():
            self.ifbb_warning_dialogue.show()
        self.sbinterface.start_diagnostic_bunch()

    def set_use_fast_kickers(self, kicker_name):
        """Set whether the fast kicker(s) should fire when the
        diagnostic bunch is fired.  To do this"""
        ksp = self.kicker_setpoint
        if self.ui.use_fast_kickers_checkbox.isChecked() and self.kicker_setpoint is not None:
            if ksp is None:
                return

            kicker_names = [k.name for k in kicker_sps]
            LOG.info(f"Enabling fast kickers for {self.screen_name}: kickers: {kicker_names}")
            # Just use first kicker name and assume that all are then
            # set (i.e they have the same kicker numbers---they should
            # be configured as such on the doocs server...).
            self.sbinterface.set_kicker_name(kicker_name)
            # Power on kickers if they are to be used.
            if not self.sbinterface.get_use_kicker():
                self.sbinterface.power_on_kickers()
        else:
            # Set kickers not to be used in the SBM.
            self.sbinterface.dont_use_kickers()

    def goto_last_bunch_in_machine(self):
        beam_regions = get_beam_regions(get_bunch_pattern())
        last_beam_region = beam_regions[-1]
        nbunches = last_beam_region.nbunches()
        beam_region_number = last_beam_region.idn
        diagnostic_bunch_number = nbunches + 1
        # assert last_beam_region > 0
        LOG.info(f"Found last bunch in machine: BR = {beam_region_number}, last normal bunch no. = {nbunches}, diagnostic bunch no. = {diagnostic_bunch_number}")
        self.sbinterface.set_beam_region(beam_region_number - 1)
        self.sbinterface.set_bunch_number(diagnostic_bunch_number)

    def goto_last_bunch_in_br(self):
        beam_regions = get_beam_regions(get_bunch_pattern())
        # This is zero counting!! beam region 1 is 0 when read from sbunch midlayer!
        selected_beam_region = self.sbinterface.get_beam_region()
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
            self.sbinterface.set_bunch_number(br.nbunches())

    # def update_panel_start_stop_state(self):
    #     is_diag_bunch_firing = self.sbinterface.is_diag_bunch_firing()
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

        self.sbinterface = sbinterface

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
            self.sbinterface.set_ibfb_state(on=False)
            time.sleep(0.1)
            self.sbinterface.start_diagnostic_bunch()
        elif button == self.ignore_and_go_button:
            self.sbinterface.start_diagnostic_bunch()
        elif button == self.cancel_button:
            self.hide()
