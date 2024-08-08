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
from esme.core import DiagnosticRegion
from esme.gui.widgets.stack import DiagnosticStack


class BeamLossMonitorStatus(QWidget):
    def __init__(self, blm: BeamLossMonitor, parent=None):
        super().__init__(parent)

        self.blm = blm
        self._init_ui()

    def _init_ui(self):
        self.layout = QHBoxLayout()

        self.name_label = QLabel(self.blm.name)
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
        blm = BeamLossMonitor(name)
        blm_widget = BeamLossMonitorStatus(blm)
        self.content_layout.addWidget(blm_widget)
        self.content_widget.setMinimumHeight(
            self.content_layout.count() * 50
        )  # Adjust height as needed

    def _update_ui(self) -> None:
        for blm in self._blm_widgets:
            blm.update_progress()


class BeamLossMonitorsStack(DiagnosticStack):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(
            parent=parent,
            i1_widget=BeamLossMonitorPanel(),
            b2_widget=BeamLossMonitorPanel(),
        )

    def add_blms(self, region: DiagnosticRegion, names: list[str]) -> None:
        blm_panel = self.get_widget_by_region(region)
        for name in names:
            blm_panel.add_blm(name)

    def add_blms_for_regions(
        self,
        i1blms: list[str] | None = None,
        b1blms: list[str] | None = None,
        b2blms: list[str] | None = None,
    ) -> None:
        if i1blms:
            self.add_blms(DiagnosticRegion.I1, i1blms)
        if b1blms:
            self.add_blms(DiagnosticRegion.B1, b1blms)
        if b2blms:
            self.add_blms(DiagnosticRegion.B2, b2blms)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    stack = BeamLossMonitorsStack()
    stack.add_blms_for_regions(
        i1blms=["BLM.59.I1", "BLM.70.I1", "AsdA"], b2blms=["asdads", "asdasd", "AsdA"]
    )
    stack.set_widget_by_region(DiagnosticRegion.I1)
    stack.show()

    sys.exit(app.exec_())
