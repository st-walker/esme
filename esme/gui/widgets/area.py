import logging
import time
import textwrap

from PyQt5.QtCore import QProcess, QTimer, pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from esme.control.exceptions import EuXFELMachineError
from esme.control.screens import Screen, screen_is_fully_operational, PoweringState
from esme.core import DiagnosticRegion
from esme.gui.ui import Ui_area_widget
from esme.control.exceptions import DOOCSReadError
from esme.gui.widgets.common import get_machine_manager_factory, raise_message_box

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
        self._screens_we_set_to_take_data: set[Screen] = set()

        self.jddd_camera_window_processes: dict[str, QProcess] = {}

        self.update_screen_combo_box(self.I1_INITIAL_SCREEN)
        self._setup_screen(self.I1_INITIAL_SCREEN)

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
        self.jddd_camera_window_processes[screen_name] = process

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
        """check the screen is on, if it's not, we try and power it.
        similarly, it might be powered, but not yet taking data/acquiring images,
        so we also try to take care of that if necessary."""
        screen = self.machine.screens[screen_name]

        try:
            is_powered = screen.is_powered()
            is_taking_data = screen.is_acquiring_images()
        except DOOCSReadError:
            # If we cannot even ask the screen if it is powered,
            # Then we can reasonably assume that the computer
            # responsible for this camera has crashed.
            # XXX: Perhaps this should be changed to a different exception
            # Raised by the screen...
            raise_message_box(
                text=(f"{screen_name}'s camera is unreachable, "
                      " perhaps the server responsible "
                      " for this camera has crashed."),
                informative_text=("Check the server status under"
                                  " Status → Miscellaneous → Camera Status"),
                title="Unreachable Screen",
                icon="Critical",
            )

        if is_powered and is_taking_data:
            return

        global _CAMERA_DIALOGUE
        _CAMERA_DIALOGUE = CameraDialogue(screen_name, 
                                          is_powered=is_powered, 
                                          is_taking_data=is_taking_data)
        _CAMERA_DIALOGUE.show()

        try:
            timeout = 10
            try_to_boot_screen(screen, timeout=timeout)
        except EuXFELMachineError:
            # Give up, couldn't turn it on...
            raise_message_box(
                text=f"Given up trying to boot screen after {timeout}s: {screen_name}",
                title="Unbootable Screen",
                icon="Warning",
            )
        else:
            if is_powered: # or was powered, i.e., it was not powered before we did it.
                self._screens_we_set_to_take_data.add(screen)
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

        self._setup_screen(screen_name)
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
        if not self._screens_we_powered | self._screens_we_set_to_take_data:
            return

        dialog = SwitchOffCamerasDialog([s.name for s in self._screens_we_powered],
                                        [s.name for s in self._screens_we_set_to_take_data])
        if dialog.exec_() == QDialog.Accepted:
            for camera in self._screens_we_powered:
                camera.power_on_off(on=False)
            for camera in self._screens_we_set_to_take_data:
                camera.start_stop_image_acquisition(acquire=False)

    def close_jddd_screen_windows(self) -> None:
        for process in self.jddd_camera_window_processes.values():
            process.close()


class CameraDialogue(QDialog):
    def __init__(self, camera_name: str, *, is_powered: bool, is_taking_data: bool, parent: QWidget | None = None):
        super().__init__(parent)

        self.setWindowTitle(f"{camera_name}'s Camera Status")
        
        if not is_powered:
            message = f"Camera {camera_name} powered off and not taking data.\nPowering and setting up camera for image acquisition..."
        elif is_powered and not is_taking_data:
            message = f"Camera {camera_name} is not taking data.\nStarting image acquisition..."
        else:
            message = "Camera status: unknown: is powered: %s, is taking data: %s" % (is_powered, is_taking_data)

        self.label = QLabel(message)

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
    def __init__(self, we_switched_on: list[str],
                we_started_taking_data: list[str]):
        super().__init__()
        self.we_switched_on = we_switched_on
        self.we_started_taking_data = we_started_taking_data
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Restore Cameras")
        layout = QVBoxLayout()

        message = textwrap.dedent("""\
        One or more cameras was powered or started taking
        data due to this application. Should the cameras
        be restored to their previous states?

        """)

        started_taking_data_bullets = ""
        for name in self.we_started_taking_data:
            started_taking_data_bullets += f"• {name}\n"

        stop_taking_data_message = textwrap.dedent(f"""\
        Would stop acquiring images:
        {started_taking_data_bullets}
        """)

        power_off_bullets = ""
        for name in self.we_switched_on:
            power_off_bullets += f"• {name}\n"
        power_off_message = textwrap.dedent(f"""\
        Would power off:
        {power_off_bullets}
        """)

        if started_taking_data_bullets:
            message += stop_taking_data_message

        if power_off_bullets:
            message += power_off_message

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
