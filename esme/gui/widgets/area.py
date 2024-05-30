import logging
import time

from PyQt5.QtCore import QProcess, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from esme.control.exceptions import EuXFELMachineError
from esme.control.screens import Screen, screen_is_fully_operational
from esme.core import DiagnosticRegion
from esme.gui.ui import Ui_area_widget
from esme.gui.widgets.common import get_machine_manager_factory

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

_CAMERA_DIALOGUE = None


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

        # Build two machine interfaces, one for I1 diagnostics and the
        # other for B2 diagnostics.
        machines = get_machine_manager_factory().make_i1_b2_managers()
        self.i1machine, self.b2machine = machines
        self.machine = self.i1machine  # Set initial machine choice to
        # be for I1 diagnostics

        self.ui.i1_radio_button.pressed.connect(self.set_i1)
        self.ui.b2_radio_button.pressed.connect(self.set_b2)
        self.ui.jddd_screen_gui_button.clicked.connect(self.open_jddd_screen_window)
        self.ui.select_screen_combobox.activated.connect(self.select_screen)
        # self.ui.select_screen_combobox.activated.connect(self.select_screen)

        # If we pick a particular screen in I1, then we click to go to B2, then back to I1,
        # it remembers which screen we are on.  Only net for I1 as we start in I1.
        self._selected_i1_screen = self.I1_INITIAL_SCREEN
        self._selected_b2_screen = None
        self._screens_we_powered: set[Screen] = set()

        self.jddd_camera_window_processes: dict[str, QProcess] = {}

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
        screen_name = self.get_selected_screen_name()
        process = QProcess()
        command = f"jddd-run -file commonAll_In_One_Camera_Expert.xml -address XFEL.DIAG/CAMERA/{screen_name}/"
        process.start(command)
        process.waitForStarted()
        self.jddd_camera_window_processes[screen_name]

    def set_i1(self) -> None:
        # This if loop is just to basically restore state so when we
        # click on I1 after being on B2, the screen name from before
        # is remembered rather than reset to some default value every time.
        if self.machine is self.b2machine:
            # Keep track of selected screen if moving from B2 to I1.
            self._selected_b2_screen = self.get_selected_screen_name()

        LOG.debug(f"Setting area in {self} to I1")
        self.machine = self.i1machine
        screen_name = self._selected_i1_screen or self.I1_INITIAL_SCREEN
        self.update_screen_combo_box(screen_name)

    def set_b2(self):
        if self.machine is self.i1machine:
            # Keep track of selected screen if moving from I1 to B2.
            self._selected_i1_screen = self.get_selected_screen_name()

        LOG.debug(f"Setting area in {self} to B2")
        self.machine = self.b2machine
        screen_name = self._selected_b2_screen or self.B2_INITIAL_SCREEN
        self.update_screen_combo_box(screen_name)

    def _setup_screen(self, screen_name: str) -> None:
        """check the screen is on, if it's not, we try and power it"""
        screen = self.machine.screens[screen_name]

        if screen_is_fully_operational(screen):
            return

        global _CAMERA_DIALOGUE
        _CAMERA_DIALOGUE = CameraDialogue(screen_name)
        _CAMERA_DIALOGUE.show()

        try:
            timeout = 10
            Camera
            try_to_boot_screen(screen, timeout=timeout)
        except EuXFELMachineError:
            # Give up, couldn't turn it on...
            raise_message_box(
                text=f"Given up trying to boot screen after {timeout}s: {screen_name}",
                title="Unbootable Screen",
                icon="Warning",
            )
        else:
            self._screens_we_powered.add(screen)

    def select_screen(self, index: int) -> None:
        screen_name: str = self.ui.select_screen_combobox.itemText(index)

        # Avoid emitting needlessly, if we are just selecting the
        # screen we are already on, then do nothing.
        if screen_name == self._selected_i1_screen:
            return
        if screen_name == self._selected_b2_screen:
            return

        self._setup_screen()
        # Emitting here can be expensive as the rest of the GUI learns from this one signal
        # Where we are in the machine (the region, I1 or B2, is inferred from the screen name.)
        self.screen_name_signal.emit(screen_name)

    def set_area(self, area: DiagnosticRegion) -> None:
        if area is self.machine.area:
            return

        if area is self.i1machine.area:
            self.set_i1()
        elif area is self.b2machine.area:
            self.set_b2()

    def closeEvent(self, event) -> None:
        self.close_jddd_screen_windows()

        dialog = SwitchOffCamerasDialog(cameras)
        if dialog.exec_() == QDialog.Accepted:
            for camera in self._screens_we_powered:
                camera.power_on_off(on=False)

    def close_jddd_screen_windows(self) -> None:
        for process in self.jddd_camera_window_processes.values():
            process.close()


class CameraDialogue(QDialog):
    def __init__(self, camera_name: str, parent: QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle("Camera Status")

        self.label = QLabel("Camera Statuspowered. Powering...")

        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.ok_button)

        self.setLayout(layout)

        # Set the dialog to be modal.
        self.setWindowModality(Qt.ApplicationModal)


def try_to_boot_screen(screen: Screen, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout

    if screen_is_fully_operational(screen):
        return

    def try_it() -> None:
        if time.time() > deadline:
            LOG.critical("Given up trying to boot %s after %s", screen.name, timeout)

        punt_time = 1000
        if screen.get_powering_state() is not PoweringState.STATIC:
            punt_time = 2000
        elif not screen.is_powered():
            screen.power_on_off(on=True)
        elif not screen.is_acquiring_images():
            screen.start_stop_image_acquisition(acquire=True)

        if not screen_is_fully_operational(screen):
            QTimer.singleShot(punt_time, try_it)

    try_it()


class SwitchOffCamerasDialog(QDialog):
    def __init__(self, screen_names: list[str]):
        super().__init__()
        self.screens = screens
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Switch Off Cameras")
        layout = QVBoxLayout()

        # Add message label
        message = "Switch off cameras switched on by this app?\n"
        for screen in self.screen_names:
            message += f"• {screen}\n"

        message_label = QLabel(message, self)
        layout.addWidget(message_label)

        # Add buttons
        button_layout = QHBoxLayout()

        yes_button = QPushButton("Yes", self)
        yes_button.clicked.connect(self.accept)
        button_layout.addWidget(yes_button)

        no_button = QPushButton("No", self)
        no_button.clicked.connect(self.reject)
        button_layout.addWidget(no_button)

        layout.addLayout(button_layout)

        self.setLayout(layout)
