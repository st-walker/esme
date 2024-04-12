from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal, QProcess

from esme.gui.ui import Ui_area_widget
from esme.gui.widgets.common import make_default_i1_lps_machine, make_default_b2_lps_machine



class AreaControl(QWidget):
    screen_name_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_area_widget()
        self.ui.setupUi(self)

        # Build two machine interfaces, one for I1 diagnostics and the other for B2 diagnostics.
        self.i1machine = make_default_i1_lps_machine()
        self.b2machine = make_default_b2_lps_machine()
        self.machine = self.i1machine # Set initial machine choice to be for I1 diagnostics

        self.ui.i1_radio_button.pressed.connect(self.set_i1)
        self.ui.b2_radio_button.pressed.connect(self.set_b2)
        self.ui.jddd_screen_gui_button.clicked.connect(self.open_jddd_screen_window)
        self.ui.select_screen_combobox.activated.connect(self.select_screen)
        
        self.update_screen_combo_box()
        
    def update_screen_combo_box(self) -> None:
        self.ui.select_screen_combobox.clear()
        self.ui.select_screen_combobox.addItems(self.machine.screens)
        
    def emit_current_screen_name(self) -> None:
        self.screen_name_signal.emit(self.get_selected_screen_name())

    def get_selected_screen_name(self) -> str:
        return self.ui.select_screen_combobox.currentText()
        
    def open_jddd_screen_window(self) -> None:
        self.jddd_camera_window_process = QtCore.QProcess()
        screen = self.get_selected_screen_name()
        command = f"jddd-run -file commonAll_In_One_Camera_Expert.xml -address XFEL.DIAG/CAMERA/{screen}/"
        self.jddd_camera_window_process.start(command)
        self.jddd_camera_window_process.waitForStarted()
        self.jddd_camera_window_process.finished.connect(self.close_jddd_screen_window)

    def set_i1(self) -> None:
        LOG.debug(f"Setting area in {self} to I1")
        self.machine = self.i1machine
        self.update_screen_combo_box()

    def set_b2(self):
        LOG.debug(f"Setting area in {self} to B2<")
        self.machine = self.b2machine
        self.update_screen_combo_box()

    def select_screen(self, _):
        screen_name = self.ui.select_screen_combobox.currentText()
        self.screen_name_signal.emit(screen_name)
        self.machine.set_kickers_for_screen(screen_name)

    def closeEvent(self, event) -> None:
        self.close_jddd_screen_window()

    def close_jddd_screen_window(self) -> None:
        try:
            self.jddd_camera_window_process.close()
        except AttributeError:
            pass
        else:
            self.jddd_camera_window_process = None
        
