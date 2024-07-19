import sys

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from esme.control.blms import BeamLossMonitor


class BeamLossMonitorStatus(QWidget):
    def __init__(self, blm: BeamLossMonitor, parent=None):
        super().__init__(parent)

        self.blm = blm
        self._init_ui()

    def _init_ui(self):
        self.layout = QHBoxLayout()

        self.name_label = QLabel(name)
        self.layout.addWidget(self.name_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(self.blm.get_slow_threshold())
        self.layout.addWidget(self.progress_bar)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.reset_progress)
        self.layout.addWidget(self.reset_button)

        self.setLayout(self.layout)

        # Timer to periodically check the callback value
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.timer.start(1000)  # Update every second

    def reset_progress(self):
        self.blm.slow_protection_reset()
        self.update_progress()

    def update_progress(self) -> None:
        value = self.callback()
        self.progress_bar.setValue(int(self.blm.get_slow_counter()))


class BeamLossMonitorPanel(QWidget):
    def __init__(self):
        super().__init__()

        self._init_ui()
        self._blm_widgets = []

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_ui)
        self._timer.start(1000)

    def _init_ui(self):
        # Main layout
        self.layout = QVBoxLayout(self)

        # Scroll area
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        # Content widget to hold the rows
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)

        self.scroll_area.setWidget(self.content_widget)

        self.layout.addWidget(self.scroll_area)

        self.setLayout(self.layout)

    def add_blm(self, name: str) -> None:
        blm_widget = BeamLossMonitorStatus(name)
        self.content_layout.addWidget(blm_widget)
        self.content_widget.setMinimumHeight(
            self.content_layout.count() * 50
        )  # Adjust height as needed

    def _update_ui(self) -> None:
        for blm in self._blm_widgets:
            blm.update_progress()


class BeamLossMonitorsStack(QStackedWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent=parent)
        self._regions: defaultdict[DiagnosticRegion, list[blms]] = defaultdict(int)

    def add_blms(region: DiagnosticRegion, names: list[str]) -> None:
        self._regions[region].extend([BeamLossMonitor(name) for name in names])

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


def sample_callback():
    # This is a sample callback function
    # In practice, this function should return a value to update the progress bar
    import random

    return random.randint(0, 100)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    main_window = ScrollAreaWidget()
    main_window.add_row("Task 1", sample_callback)
    main_window.add_row("Task 2", sample_callback)

    main_window.show()

    sys.exit(app.exec_())
