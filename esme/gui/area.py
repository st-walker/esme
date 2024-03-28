from PyQt5.QtWidgets import QWidget

from esme.gui.ui import Ui_area_widget


class AreaControl(QtWidgets.QWidget):
    screen_name_signal = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.ui = Ui_area_widget(self)

    def update_screen_combo_box(self):
        self.ui.select_screen_combobox.clear()
        self.ui.select_screen_combobox.addItems(self.machine.screens)
        
    def emit_current_screen_name(self):
        self.screen_name_signal.emit(self.get_selected_screen_name())

    def get_selected_screen_name(self) -> str:
        return self.ui.select_screen_combobox.currentText()
        
    def open_jddd_screen_window(self):
        self.jddd_camera_window_process = QtCore.QProcess()
        screen = self.get_selected_screen_name()
        command = f"jddd-run -file commonAll_In_One_Camera_Expert.xml -address XFEL.DIAG/CAMERA/{screen}/"
        self.jddd_camera_window_process.start(command)
        self.jddd_camera_window_process.waitForStarted()
        self.jddd_camera_window_process.finished.connect(self.close_jddd_screen_window)
