import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget

from esme.control.target import TargetDefinition
from esme.core import DiagnosticRegion
from esme.gui.ui.target import Ui_Target
from esme.gui.widgets.sequence import TaskomatSequenceDisplay
from esme.gui.widgets.stack import DiagnosticStack


class TargetControl(QWidget):
    def __init__(
        self, parent: QWidget | None = None, dumpdefn: TargetDefinition = None
    ) -> None:
        super().__init__()

        self.ui = Ui_Target()
        self.ui.setupUi(self)
        self._connect_buttons()

        self.targ_def = dumpdefn

        self._sequence_widget = None

    def _connect_buttons(self) -> None:
        self.ui.go_to_dump_button.clicked.connect(self.go_to_dump)
        self.ui.go_straight_button.clicked.connect(self.undo_dump_and_go_straight)
        self.ui.enable_dump_control_checkbox.stateChanged.connect(self.toggle_buttons)

    def toggle_buttons(self, state: Qt.CheckState) -> None:
        self.ui.button_dump.setEnabled(state == Qt.Checked)
        self.ui.button_straight.setEnabled(state == Qt.Checked)

    def _run_init_step(self) -> None:
        self.targ_def.sequence.run_step(self.targ_def.init_step)

    def go_to_dump(self) -> None:
        self._run_init_step()
        self.targ_def.sequence.set_dynamic_property(*self.targ_def.property_to_dump)
        self._make_taskomat_window()

    def undo_dump_and_go_straight(self) -> None:
        self._run_init_step()
        self.targ_def.sequence.set_dynamic_property(*self.targ_def.property_undo_dump)
        self._make_taskomat_window()

    def _make_taskomat_window(self) -> None:
        self._sequence_widget = TaskomatSequenceDisplay(self.targ_def.sequence)
        self._sequence_widget.setWindowModality(Qt.ApplicationModal)
        self._sequence_widget.show()


class TargetStack(DiagnosticStack):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            parent=parent, i1_widget=TargetControl(), b2_widget=TargetControl()
        )

    def set_target_widget(
        self, region: DiagnosticRegion, targ_def: TargetDefinition
    ) -> None:
        tcw = self.get_widget_by_region(region)
        tcw.targ_def = targ_def


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TargetControl()
    window.setWindowTitle("QStackedWidget Example")
    window.resize(300, 200)
    window.show()
    sys.exit(app.exec_())
