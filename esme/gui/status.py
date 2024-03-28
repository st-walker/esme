import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QBrush

from esme.gui.common import make_default_i1_lps_machine, make_default_b2_lps_machine, make_i1_watcher, make_b2_watcher
from esme.control.mstate import AreaWatcher
from esme.core import region_from_screen_name
from esme import DiagnosticRegion
from esme.control.mstate import Status


class CircleIndicator(QWidget):
    COLOUR_MAPPING = {Status.GOOD: QColor(148, 255, 67),
                      Status.WARNING: QColor(220, 152, 56),
                      Status.BAD: QColor(227, 49, 30),
                      Status.UNKNOWN: QColor(63, 63, 63)}

    def __init__(self, diameter=20, tooltip_text=None, parent=None):
        super().__init__(parent)
        self.diameter = diameter
        self.status: Status = Status.UNKNOWN
        self.setFixedSize(self.diameter, self.diameter)
        if tooltip_text:
            self.setToolTip(tooltip_text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = self.COLOUR_MAPPING[self.status]
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.diameter, self.diameter)

    def set_status(self, status, tooltip_text):
        self.status = status
        self.setToolTip(tooltip_text)
        self.update()

class TextIndicator(QWidget):
    def __init__(self, true_text: str = "OPEN", false_text: str = "CLOSED", parent: QWidget = None):
        """Initialize the text indicator with custom text for true and false states.

        Args:
            true_text: The text to display when the status is True.
            false_text: The text to display when the status is False.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.true_text = true_text
        self.false_text = false_text
        self.status_true = True  # Start with true_text
        self.layout = QVBoxLayout()
        self.label = QLabel(self.true_text)
        self.label.setAlignment(Qt.AlignCenter)
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        self.update_text()

    def set_status(self, status_true: bool):
        """Update the indicator's status and displayed text.

        Args:
            status_true: The new status of the indicator.
        """
        self.status_true = status_true
        self.update_text()

    def update_text(self):
        """Update the text displayed by the indicator based on the current status."""
        self.label.setText(self.true_text if self.status_true else self.false_text)
        self.label.setStyleSheet("QLabel { font-weight: bold; font-variant: small-caps; }")

class IndicatorPanelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        self.layout.setVerticalSpacing(10)  # Reduce vertical spacing between rows
        self.indicators = []  # Store tuples of (container_widget, check_callback)

    def create_container(self, label_text: str):
        """Create a container widget with a label and return the layout."""
        container = QWidget()
        container_layout = QHBoxLayout()
        container.setLayout(container_layout)
        container_layout.setContentsMargins(0, 0, 0, 0)  # Reduce container margins

        # label = QLabel(label_text)
        # label.setAlignment(Qt.AlignVCenter)
        # container_layout.addWidget(label, alignment=Qt.AlignLeft)

        return container, container_layout

    def add_indicator(self, label_text: str, tooltip_text: str = "", check_callback=None):
        """Add a new color-based status indicator with a label to the panel."""
        container, container_layout = self.create_container(label_text)

        indicator = CircleIndicator()
        indicator.setToolTip(tooltip_text)
        label = QLabel(label_text)

        label.setAlignment(Qt.AlignVCenter)
        # indicator.setAlignment(Qt.AlignVCenter)
        row = self.layout.rowCount()
        self.layout.addWidget(label, row, 0)
        self.layout.addWidget(indicator, row, 1, alignment=Qt.AlignHCenter)

        self.indicators.append((indicator, check_callback))

    def add_text_indicator(self, label_text: str, true_text: str, false_text: str, check_callback=None):
        """Add a new text-based status indicator with a label to the panel."""
        # container, container_layout = self.create_container(label_text)

        row = self.layout.rowCount()

        text_indicator = TextIndicator(true_text=true_text, false_text=false_text)
        self.layout.addWidget(text_indicator, row, 1, alignment=Qt.AlignHCenter)

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignVCenter)
        self.layout.addWidget(label, row, 0)

        if check_callback:
            result = check_callback()
            try:
                status_true, tooltip_text = result
            except TypeError:
                status_true = result
                tooltip_text = ""
            text_indicator.set_status(status_true)
            text_indicator.setToolTip(tooltip_text)

        self.indicators.append((text_indicator, check_callback))

    def check_indicators(self):
        """Check the status of all indicators using their associated callback functions."""
        for indicator, callback in self.indicators:
            if callback:
                result = callback()
                try:
                    status_true, tooltip_text = result
                except TypeError:
                    status_true = result
                    tooltip_text = ""
                try:
                    indicator.set_status(status_true, tooltip_text)
                except:
                    indicator.set_status(status_true)
#                indicator.setToolTip(tooltip_text)


class LPSStateWatcher(IndicatorPanelWidget):
    def __init__(self, parent=None):
        super().__init__(parent=None)
        self._i1state: AreaWatcher = make_i1_watcher()
        self._b2state: AreaWatcher = make_b2_watcher()
        self.mstate: AreaWatcher = self._i1state

        self.add_text_indicator("Laser Heater Shutter", "OPEN", "CLOSED", self.mstate.is_laser_heater_shutter_open)
        self.add_text_indicator("IBFB", "OFF", "ON", self.mstate.check_ibfb_state)
        self.add_indicator("Screen", check_callback=self._check_screen_state)
        self.add_indicator("TDS", check_callback=self._check_tds_state)
        self.add_indicator("Fast Kickers", check_callback=self._check_kickers_state)

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_indicators)
        self.timer.start(1000)

    def _check_screen_state(self) -> tuple[bool, str]:
        return self.mstate.check_screen_state()

    def _check_tds_state(self) -> tuple[bool, str]:
        return self.mstate.check_tds_state()

    def _check_kickers_state(self) -> tuple[bool, str]:
        return self.mstate.check_tds_state()

    def set_screen(self, screen: str) -> None:
        region = region_from_screen_name(screen)
        if region is DiagnosticRegion.I1:
            self.mstate = self._i1state
        elif region is DiagnosticRegion.B2:
            self.mstate = self._b2state
        else:
            raise ValueError("Unexpected Diagnostic Region")
        self.mstate.watched_screen_name = screen



# Boilerplate code to initialize and run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    panel = IndicatorPanelWidget()

    # Example usage
    panel.add_indicator("Connectivity", "Indicator tooltip", lambda: True)
    panel.add_text_indicator("Server Status", "ONLINE", "OFFLINE", lambda: (False, "Server is currently offline"))

    panel.show()
    sys.exit(app.exec_())
