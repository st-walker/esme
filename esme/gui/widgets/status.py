import sys
from typing import Callable

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QHBoxLayout
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QBrush

from .common import get_machine_manager_factory, make_i1_watcher, make_b2_watcher
from esme.control.mstate import AreaWatcher
from esme.core import region_from_screen_name
from esme import DiagnosticRegion
from esme.control.mstate import Health, Condition


class CircleIndicator(QWidget):
    COLOUR_MAPPING = {Health.GOOD: QColor(148, 255, 67), # Green
                      Health.WARNING: QColor(220, 152, 56), # Orange
                      Health.BAD: QColor(227, 49, 30), # Red
                      Health.UNKNOWN: QColor(63, 63, 63), # Grey
                      Health.SUBJECTIVE: QColor(16, 62, 180) # Blue
                      }

    def __init__(self, diameter: int = 20, tooltip_text: str | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.diameter = diameter
        self.status: Health = Health.UNKNOWN
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

    def set_status(self, condition: Condition) -> None:
        self.status = condition.health
        self.setToolTip(condition.long)
        self.update()

class TextIndicator(QWidget):
    def __init__(self, initial_state: str = "", parent: QWidget | None = None):
        """Initialize the text indicator with custom text for true and false states.

        Args:
            true_text: The text to display when the status is True.
            false_text: The text to display when the status is False.
            parent: The parent widget.
        """
        super().__init__(parent)
        self.label = QLabel()
        self.wlayout = QVBoxLayout()        
        self.label.setAlignment(Qt.AlignCenter) # type: ignore
        self.wlayout.addWidget(self.label)
        self.setLayout(self.wlayout)
        self.label.setStyleSheet("QLabel { font-weight: bold; font-variant: small-caps; }")

    def set_status(self, condition: Condition) -> None:
        """Update the text displayed by the indicator based on the current status."""
        self.label.setText(condition.short)
        self.label.setToolTip(condition.long)


class IndicatorPanelWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.wlayout = QGridLayout()
        self.setLayout(self.wlayout)
        self.wlayout.setVerticalSpacing(10)  # Reduce vertical spacing between rows

        # Store tuples of (container_widget, check_callback)
        self.indicators: list[tuple[TextIndicator | CircleIndicator, Callable[[], Condition]]] = []

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

    def add_indicator(self, label_text: str, check_callback: Callable[[], Condition]):
        pass
    
    def add_state_indicator(self, label_text: str, check_callback: Callable[[], Condition]):
        """Add a new color-based status indicator with a label to the panel."""
        container, container_layout = self.create_container(label_text)

        indicator = CircleIndicator()
        # indicator.setToolTip(tooltip_text)
        label = QLabel(label_text)

        label.setAlignment(Qt.AlignVCenter) # type: ignore
        # indicator.setAlignment(Qt.AlignVCenter)
        row = self.wlayout.rowCount()
        self.wlayout.addWidget(label, row, 0)
        self.wlayout.addWidget(indicator, row, 1, alignment=Qt.AlignHCenter) # type: ignore

        self.indicators.append((indicator, check_callback))

    def add_text_indicator(self, label_text: str,
                           check_callback: Callable[[], Condition]) -> None:
        """Add a new text-based status indicator with a label to the panel."""
        # container, container_layout = self.create_container(label_text)

        row = self.wlayout.rowCount()

        text_indicator = TextIndicator(initial_state="")
        self.wlayout.addWidget(text_indicator, row, 1, alignment=Qt.AlignHCenter) # type: ignore

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignVCenter)
        self.wlayout.addWidget(label, row, 0)

        self.indicators.append((text_indicator, check_callback))

    def check_indicators(self):
        """Check the status of all indicators using their associated callback functions."""
        for indicator, callback in self.indicators:
            indicator.set_status(callback())


class LPSStateWatcher(IndicatorPanelWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=None)
        self._i1state: AreaWatcher = make_i1_watcher()
        self._b2state: AreaWatcher = make_b2_watcher()
        self.mstate: AreaWatcher = self._i1state

        self.add_text_indicator("Laser Heater Shutter", self.mstate.get_laser_heater_shutter_state)
        self.add_text_indicator("IBFB", self.mstate.get_ibfb_state)

        self.add_state_indicator("Screen", check_callback=self._check_screen_state)
        self.add_state_indicator("TDS", check_callback=self._check_tds_state)
        self.add_state_indicator("Fast Kickers", check_callback=self._check_kickers_state)

        self.timer = QTimer()
        self.timer.timeout.connect(self.check_indicators)
        self.timer.start(500)

    def _check_screen_state(self) -> Condition:
        return self.mstate.check_screen_state()

    def _check_tds_state(self) -> Condition:
        return self.mstate.check_tds_state()

    def _check_kickers_state(self) -> Condition:
        return self.mstate.check_kickers_state()

    def set_screen(self, screen: str) -> None:
        region = region_from_screen_name(screen)
        if region is DiagnosticRegion.I1:
            self.mstate = self._i1state
        elif region is DiagnosticRegion.B2:
            self.mstate = self._b2state
        else:
            raise ValueError("Unexpected Diagnostic Region")
        self.mstate.watched_screen_name = screen
