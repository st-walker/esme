# from PyQt5.QtCore import QPointF
# from PyQt5.QtGui import QColor, QPainter, QBrush
# from PyQt5.QtWidgets import QAbstractButton, QPushButton, QCheckBox,
from importlib_resources import files
import logging

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot

from .ui.special_bunch_panel import Ui_special_bunch_panel
from esme.control.configs import build_simple_machine_from_config

LOG = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = files("esme.gui") / "defaultconf.yml"


class SpecialBunchControl(QtWidgets.QWidget):
    def __init__(self, parent=None, machine=None):
        super().__init__(parent=parent)

        if machine is None:
            self.machine = build_simple_machine_from_config(DEFAULT_CONFIG_PATH)
        else:
            self.machine = machine


        self.ui = Ui_special_bunch_panel()
        self.ui.setupUi(self)

        self.connect_buttons()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: None)
        self.timer.timeout.connect(self.update)
        self.timer.start(100)

    def update(self):
        self.update_screen_combo_box()
        self.ui.beamregion_spinbox.setValue(self.machine.sbunches.get_beam_region() + 1)
        self.ui.bunch_spinbox.setValue(self.machine.sbunches.get_bunch_number())      

    def set_bunch_control_enabled(self, enabled):
        self.ui.beamregion_spinbox.setEnabled(enabled)
        self.ui.bunch_spinbox.setEnabled(enabled)
        self.ui.go_to_last_bunch_in_br_pushbutton.setEnabled(enabled)
        self.ui.go_to_last_laserpulse_pushbutton.setEnabled(enabled)
        self.ui.use_fast_kickers_checkbox.setEnabled(enabled)
        self.ui.use_tds_checkbox.setEnabled(enabled)        
        self.ui.select_screen_combobox.setEnabled(enabled)
        self.ui.npulses_spinbox.setEnabled(enabled)

    def connect_buttons(self):
        self.ui.select_screen_combobox.currentIndexChanged.connect(self.configure_kickers)
        self.ui.use_fast_kickers_checkbox.stateChanged.connect(self.set_use_fast_kickers)
        self.ui.beamregion_spinbox.valueChanged.connect(lambda n: self.machine.sbunches.set_beam_region(n - 1))
        self.ui.bunch_spinbox.valueChanged.connect(self.machine.sbunches.set_bunch_number)
        self.ui.npulses_spinbox.valueChanged.connect(self.machine.sbunches.set_npulses)
        self.ui.go_to_last_laserpulse_pushbutton.clicked.connect(self.goto_last_bunch_in_machine)
        self.ui.go_to_last_bunch_in_br_pushbutton.clicked.connect(self.goto_last_bunch_in_br)
        self.ui.start_stop_button.clicked.connect(self.start_stop_special_bunches)

    def update_screen_combo_box(self):
        self.ui.select_screen_combobox.clear()
        for screen_name in self.machine.screens.active_region_screen_names():
            self.ui.select_screen_combobox.addItem(screen_name)
        
    def configure_kickers(self):
        # self.clear_image()
        LOG.info(f"Configuring kickers for screen: {self.get_selected_screen_name()}")
        self.machine.set_kicker_for_screen(self.get_selected_screen_name())
        self.screen_worker.screen_name = self.get_selected_screen_name()

    def set_use_fast_kickers(self):
        if self.ui.use_fast_kickers_checkbox.isChecked():
            screen_name = self.get_selected_screen_name()
            kicker_sps = self.machine.screens.get_fast_kicker_setpoints_for_screen(screen_name)
            kicker_names = [k.name for k in kicker_sps]
            LOG.info(f"Enabling fast kickers for {screen_name}: kickers: {kicker_names}")
            # Just use first one and assume they are the same (they should be
            # configured as such on the doocs server...)
            self.machine.sbunches.set_kicker(kicker_names[0])
        else:
            self.machine.sbunches.set_dont_use_fast_kickers()

    def goto_last_bunch_in_machine(self):
        beam_regions = get_beam_regions(get_bunch_pattern())
        last_beam_region = beam_regions[-1]
        nbunches = last_beam_region.nbunches()
        beam_region_number = last_beam_region.idn
        diagnostic_bunch_number = nbunches + 1
        # assert last_beam_region > 0
        LOG.info(f"Found last bunch in machine: BR = {beam_region_number}, last normal bunch no. = {nbunches}, diagnostic bunch no. = {diagnostic_bunch_number}")
        self.machine.sbunches.set_beam_region(beam_region_number - 1)
        self.machine.sbunches.set_bunch_number(diagnostic_bunch_number)

    def goto_last_bunch_in_br(self):
        beam_regions = get_beam_regions(get_bunch_pattern())
        # This is zero counting!! beam region 1 is 0 when read from sbunch midlayer!
        selected_beam_region = self.machine.sbunches.get_beam_region()
        assert selected_beam_region >= 0
        try:
            br = beam_regions[selected_beam_region]
        except IndexError:
            LOG.info(f"User tried to select last bunch of nonexistent beam region: {selected_beam_region}.")
            box = QMessageBox(self) #, "Invalid Beam Region", 
            box.setText(f"Beam Region {selected_beam_region} does not exist.")
            box.exec()
            return
        else:
            self.machine.sbunches.set_bunch_number(br.nbunches())
            
    @pyqtSlot()
    def start_stop_special_bunches(self):
        checked = self.ui.start_stop_button.isChecked()
        self.set_bunch_control_enabled(not checked)
        if checked:
            self.machine.sbunches.start_diagnostic_bunch()
            self.ui.start_stop_button.setText("Stop")
        else:
            self.machine.sbunches.stop_diagnostic_bunch()
            self.ui.start_stop_button.setText("Start")

    def get_selected_screen_name(self):
        return self.ui.select_screen_combobox.currentText()
            
