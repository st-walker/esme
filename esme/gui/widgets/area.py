import logging

from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QProcess

from esme.gui.ui import Ui_area_widget
from esme.gui.widgets.common import get_machine_manager_factory
from esme.core import DiagnosticRegion

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.DEBUG)


class AreaControl(QWidget):
    screen_name_signal = pyqtSignal(str)
    # These default screen names are from the Bolko tool.  I guess
    # they're the best to use for reasons of phase advance or
    # whatever.
    I1_INITIAL_SCREEN = "OTRC.58.I1"
    B2_INITIAL_SCREEN = "OTRB.457.B2"

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_area_widget()
        self.ui.setupUi(self)

        # Build two machine interfaces, one for I1 diagnostics and the other for B2 diagnostics.
        self.i1machine, self.b2machine = get_machine_manager_factory().make_i1_b2_managers()
        self.machine = self.i1machine # Set initial machine choice to be for I1 diagnostics

        self.ui.i1_radio_button.pressed.connect(self.set_i1)
        self.ui.b2_radio_button.pressed.connect(self.set_b2)
        self.ui.jddd_screen_gui_button.clicked.connect(self.open_jddd_screen_window)
        self.ui.select_screen_combobox.activated.connect(self.select_screen)
        #self.ui.select_screen_combobox.activated.connect(self.select_screen)

        # If we pick a particular screen in I1, then we click to go to B2, then back to I1,
        # it remembers which screen we are on.
        self._selected_i1_screen = self.I1_INITIAL_SCREEN
        self._selected_b2_screen = None

        self.update_screen_combo_box(self.I1_INITIAL_SCREEN)

    def update_screen_combo_box(self, initial_value=None) -> None:
        self.ui.select_screen_combobox.clear()
        self.ui.select_screen_combobox.addItems(self.machine.screens)
        if initial_value:
            index = self.ui.select_screen_combobox.findText(initial_value)
            self.ui.select_screen_combobox.setCurrentIndex(index)

    def emit_current_screen_name(self) -> None:
        self.screen_name_signal.emit(self.get_selected_screen_name())

    def get_selected_screen_name(self) -> str:
        return self.ui.select_screen_combobox.currentText()

    def open_jddd_screen_window(self) -> None:
        self.jddd_camera_window_process = QProcess()
        screen = self.get_selected_screen_name()
        command = f"jddd-run -file commonAll_In_One_Camera_Expert.xml -address XFEL.DIAG/CAMERA/{screen}/"
        self.jddd_camera_window_process.start(command)
        self.jddd_camera_window_process.waitForStarted()
        self.jddd_camera_window_process.finished.connect(self.close_jddd_screen_window)

    def set_i1(self) -> None:
        # This if loop is just to basically restore state so when we
        # click on I1 after being on B2, the screen name from before
        # is remembered rather than reset to some default value every time.
        if self.machine is self.b2machine: # Keep track of selected screen if moving from B2 to I1
            self._selected_b2_screen = self.get_selected_screen_name()

        LOG.debug(f"Setting area in {self} to I1")
        self.machine = self.i1machine
        screen_name = self._selected_i1_screen or self.I1_INITIAL_SCREEN
        self.update_screen_combo_box(screen_name)

    def set_b2(self):
        if self.machine is self.i1machine:
            self._selected_i1_screen = self.get_selected_screen_name()

        LOG.debug(f"Setting area in {self} to B2")
        self.machine = self.b2machine
        screen_name = self._selected_b2_screen or self.B2_INITIAL_SCREEN
        self.update_screen_combo_box(screen_name)

    def select_screen(self, index: int) -> None:
        screen_name: str = self.ui.select_screen_combobox.itemText(index)
        # Avoid emitting needlessly, if we have just selected the very same screen
        # as the one we started on, then do nothing.
        if screen_name == self._selected_i1_screen or screen_name == self._selected_b2_screen:
            return
        # Emitting here can be expensive as the rest of the GUI learns from this one signal
        # Where we are in the machine (the region, I1 or B2, is inferred from the screen name.)
        self.screen_name_signal.emit(screen_name)

    def set_area(self, area: DiagnosticRegion):
        if area is self.machine.area:
            return

        if area is self.i1machine.area:
            self.set_i1()
        elif area is self.b2machine.area:
            self.set_b2()

    def closeEvent(self, event) -> None:
        self.close_jddd_screen_window()

    def close_jddd_screen_window(self) -> None:
        try:
            self.jddd_camera_window_process.close()
        except AttributeError:
            pass
        else:
            self.jddd_camera_window_process = None
