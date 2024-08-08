import logging
import time

from PyQt5.QtCore import QProcess, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import QWidget

from esme.control.exceptions import DOOCSReadError, EuXFELMachineError
from esme.control.screens import (
    Position,
    PoweringState,
    Screen,
    screen_is_fully_operational,
)
from esme.core import DiagnosticRegion, region_from_screen_name
from esme.gui.ui import Ui_area_widget
from esme.gui.widgets.common import get_machine_manager_factory, raise_message_box

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

_CAMERA_DIALOGUE = None


class AreaControl(QWidget):
    screen_name_signal = pyqtSignal(str)
    region_signal = pyqtSignal(DiagnosticRegion)
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

        self._connect_buttons()

        self._selected_i1_screen = self.I1_INITIAL_SCREEN
        self._selected_b2_screen = self.B2_INITIAL_SCREEN

        self._set_keep_screens_on = False

        self.jddd_camera_window_processes: dict[str, QProcess] = {}

        self.update_screen_combo_box(self.I1_INITIAL_SCREEN)
        self._setup_screen(self.I1_INITIAL_SCREEN)

        self.timer = QTimer()
        self.timer.timeout.connect(self._update_screen_position_ui)
        self.timer.start(1000)

    def _connect_buttons(self):
        self.ui.i1_radio_button.pressed.connect(self.set_i1)
        self.ui.b2_radio_button.pressed.connect(self.set_b2)
        self.ui.jddd_screen_gui_button.clicked.connect(self.open_jddd_screen_window)
        self.ui.select_screen_combobox.activated.connect(self.select_screen)
        self.ui.screen_on_axis_button.clicked.connect(
            lambda: self._set_screen_position(Position.ONAXIS)
        )
        self.ui.screen_off_axis_button.clicked.connect(
            lambda: self._set_screen_position(Position.OFFAXIS)
        )
        self.ui.screen_out_button.clicked.connect(
            lambda: self._set_screen_position(Position.OUT)
        )
        self.ui.keep_screens_on_checkbox.stateChanged.connect(self._set_keep_screens_on)

    def _set_keep_screens_on(self, state: Qt.CheckState) -> None:
        self._set_keep_screens_on = bool(state)

    def _set_screen_position(self, pos: Position) -> None:
        screen = self.machine.screens[self.get_selected_screen_name()]
        screen.set_position(pos)

    def _update_screen_position_ui(self):
        screen = self.machine.screens[self.get_selected_screen_name()]
        self.ui.screen_position_label.setText(screen.get_position().name)

        # self.ui.select_screen_combobox.activated.connect(self.select_screen)

    def update_screen_combo_box(self, initial_value=None) -> None:
        self.ui.select_screen_combobox.clear()
        self.ui.select_screen_combobox.addItems(self.machine.screens)
        if initial_value:
            index = self.ui.select_screen_combobox.findText(initial_value)
            self.ui.select_screen_combobox.setCurrentIndex(index)

    def emit_current_screen_name(self) -> None:
        screen_name = self.get_selected_screen_name()
        area = region_from_screen_name(self.get_selected_screen_name())
        self.screen_name_signal.emit(screen_name)
        self.region_signal.emit(area)

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
            self.stop_image_acquisition_for_screen(self._selected_b2_screen)

        LOG.debug(f"Setting area in {self} to I1")
        self.machine = self.i1machine
        screen_name = self._selected_i1_screen or self.I1_INITIAL_SCREEN
        self.update_screen_combo_box(screen_name)
        self.region_signal.emit(DiagnosticRegion.I1)

    def stop_image_acquisition_for_screen(self, screen_name: str):
        if self.ui.keep_screens_on_checkbox.isChecked():
            return
        try:
            self.i1machine.screens[screen_name].start_stop_image_acquisition(
                acquire=False
            )
        except KeyError:
            self.b2machine.screens[screen_name].start_stop_image_acquisition(
                acquire=False
            )

    def set_b2(self):
        if self.machine is self.i1machine:
            # Keep track of selected screen if moving from I1 to B2.
            self._selected_i1_screen = self.get_selected_screen_name()

        LOG.debug(f"Setting area in {self} to B2")
        self.machine = self.b2machine
        screen_name = self._selected_b2_screen or self.B2_INITIAL_SCREEN
        self.update_screen_combo_box(screen_name)
        self.screen_name_signal.emit(screen_name)
        self.region_signal.emit(DiagnosticRegion.B2)

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
                text=(
                    f"{screen_name}'s camera is unreachable, "
                    " perhaps the server responsible "
                    " for this camera has crashed."
                ),
                informative_text=(
                    "Check the server status under"
                    " Status → Miscellaneous → Camera Status"
                ),
                title="Unreachable Screen",
                icon="Critical",
            )
            return

        if is_powered and is_taking_data:
            return

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

    def select_screen(self, index: int) -> None:
        screen_name: str = self.ui.select_screen_combobox.itemText(index)

        self._setup_screen(screen_name)
        # Emitting here can be expensive as the rest of the GUI learns from this one signal
        # Where we are in the machine (the region, I1 or B2, is inferred from the screen name.)
        self.screen_name_signal.emit(screen_name)
        if self.i1machine is self.machine:
            self.stop_image_acquisition_for_screen(self._selected_i1_screen)
            self._selected_i1_screen = screen_name
        elif self.b2machine is self.machine:
            self.stop_image_acquisition_for_screen(self._selected_b2_screen)
            self._selected_b2_screen = screen_name

    def set_area(self, area: DiagnosticRegion) -> None:
        if area is self.machine.area:
            return

        if area is self.i1machine.area:
            self.set_i1()
        elif area is self.b2machine.area:
            self.set_b2()

    def closeEvent(self, event) -> None:
        for process in self.jddd_camera_window_processes.values():
            process.close()


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
