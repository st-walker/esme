import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QStackedWidget, QWidget

from esme.gui.ui.target import Ui_Target
from esme.core import DiagnosticRegion
from esme.gui.widgets.sequence import TaskomatSequenceDisplay
from esme.control.taskomat import Sequence


class TargetControl(QWidget):
    def __init__(
        self,
        dump_name: str,
        sequence_dump: Sequence,
        sequence_undo: Sequence,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__()

        self.ui = Ui_Target()
        self.ui.setupUi(self)
        self._connect_buttons()

        self.sequence_dump = sequence_dump
        self.sequence_undo = sequence_undo

        self._sequence_widget = None

    def _connect_buttons(self) -> None:
        self.ui.go_to_dump_button.clicked.connect(self.go_to_dump)
        self.ui.go_straight_button.clicked.connect(self.go_straight)
        self.ui.enable_panel_checkbox.stateChanged.connect(self.toggle_buttons)

    def toggle_buttons(self, state: Qt.CheckState) -> None:
        self.ui.button_dump.setEnabled(state == Qt.Checked)
        self.ui.button_straight.setEnabled(state == Qt.Checked)

    def go_to_dump(self) -> None:
        self._sequence_widget = TaskomatSequenceDisplay(self.sequence_dump)
        self._sequence_widget.setWindowModality(Qt.ApplicationModal)
        self._sequence_widget.show()

    def undo_dump_and_go_straight(self) -> None:
        self._sequence_widget = TaskomatSequenceDisplay(self.sequence_undo)
        self._sequence_widget.setWindowModality(Qt.ApplicationModal)
        self._sequence_widget.show()


class TargetStack(QStackedWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self._regions: list[DiagnosticRegion] = []

    def add_target_widget(
        self,
        region: DiagnosticRegion,
        dump_name: str,
        dump_sequence: Sequence,
        undo_dump_sequence: Sequence,
    ) -> None:
        tc = TargetStack(dump_name, dump_sequence, undo_dump_sequence)
        self._stacked_widget.addWidget(tc)
        self._keys.append(key)

    def set_region(self, section: DiagnosticRegion) -> None:
        self.stacked_widget.setCurrentIndex(self._regions[section])


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TargetControl()
    window.setWindowTitle("QStackedWidget Example")
    window.resize(300, 200)
    window.show()
    sys.exit(app.exec_())
