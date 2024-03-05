import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QGridLayout, QHBoxLayout
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QPainter, QColor, QBrush

class CircleIndicator(QWidget):
    def __init__(self, diameter=20, tooltip_text=None, parent=None):
        super().__init__(parent)
        self.diameter = diameter
        self.status_ok = True
        self.setFixedSize(self.diameter, self.diameter)
        if tooltip_text:
            self.setToolTip(tooltip_text)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor(148, 255, 67) if self.status_ok else QColor(227, 49, 30)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, self.diameter, self.diameter)

    def update_status(self, status_ok, tooltip_text):
        self.status_ok = status_ok
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

        label = QLabel(label_text)
        label.setAlignment(Qt.AlignVCenter)
        container_layout.addWidget(label, alignment=Qt.AlignLeft)

        return container, container_layout

    def add_indicator(self, label_text: str, tooltip_text: str = "", check_callback=None):
        """Add a new color-based status indicator with a label to the panel."""
        container, container_layout = self.create_container(label_text)

        indicator = CircleIndicator()
        indicator.setToolTip(tooltip_text)
        container_layout.addWidget(indicator, 1, alignment=Qt.AlignHCenter)

        row = self.layout.rowCount()
        self.layout.addWidget(container, row, 0)

        self.indicators.append((indicator, check_callback))

    def add_text_indicator(self, label_text: str, true_text: str, false_text: str, check_callback=None):
        """Add a new text-based status indicator with a label to the panel."""
        container, container_layout = self.create_container(label_text)

        text_indicator = TextIndicator(true_text=true_text, false_text=false_text)
        container_layout.addWidget(text_indicator, 1, alignment=Qt.AlignHCenter)

        row = self.layout.rowCount()
        self.layout.addWidget(container, row, 0)

        if check_callback:
            status_true, tooltip_text = check_callback()
            text_indicator.set_status(status_true)
            text_indicator.setToolTip(tooltip_text)

        self.indicators.append((text_indicator, check_callback))

    def check_indicators(self):
        """Check the status of all indicators using their associated callback functions."""
        for indicator, callback in self.indicators:
            if callback:
                result = callback()
                if isinstance(result, tuple):  # For TextIndicator
                    status_true, tooltip_text = result
                    indicator.set_status(status_true)
                    indicator.setToolTip(tooltip_text)
                else:  # For CircleIndicator or similar
                    indicator.set_status(result)


class LPSStateWatcher(IndicatorPanelWidget):
    pass


# Boilerplate code to initialize and run the application
if __name__ == '__main__':
    app = QApplication(sys.argv)
    panel = IndicatorPanelWidget()

    # Example usage
    panel.add_indicator("Connectivity", "Indicator tooltip", lambda: True)
    panel.add_text_indicator("Server Status", "ONLINE", "OFFLINE", lambda: (False, "Server is currently offline"))

    panel.show()
    sys.exit(app.exec_())
