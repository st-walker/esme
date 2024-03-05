# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,
import logging

from PyQt5 import QtWidgets
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QMessageBox

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

        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(100)

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
        self.update_panel_start_stop_state()

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
        self.ui.start_button.clicked.connect(self.sbinterface.start_diagnostic_bunch)
        self.ui.stop_button.clicked.connect(self.sbinterface.stop_diagnostic_bunch)

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

    def update_panel_start_stop_state(self):
        is_diag_bunch_firing = self.sbinterface.is_diag_bunch_firing()
        if is_diag_bunch_firing:
            self.set_bunch_control_enabled(False)
            self.ui.start_button.setEnabled(False)
            self.ui.stop_button.setEnabled(True)
        else:
            self.set_bunch_control_enabled(True)
            self.ui.stop_button.setEnabled(True)
            self.ui.stop_button.setEnabled(False)
