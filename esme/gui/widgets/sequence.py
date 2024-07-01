from types import SimpleNamespace

from PyQt5.QtCore import QProcess, Qt
from PyQt5.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen
from PyQt5.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from esme.control.taskomat import Sequence


class StrikethroughLabel(QLabel):
    def __init__(self, text: str = "", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.strikethrough = False

    def set_strikethrough(self, enabled: bool) -> None:
        self.strikethrough = enabled
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        super().paintEvent(event)
        if self.strikethrough:
            painter = QPainter(self)
            pen = QPen(
                self.palette().text().color(), 2
            )  # Adjust the second parameter for thicker line
            painter.setPen(pen)
            metrics = self.fontMetrics()
            y = self.height() // 2
            painter.drawLine(0, y, metrics.width(self.text()), y)


class TaskomatSequenceDisplay(QWidget):
    def __init__(self, sequence: Sequence) -> None:
        super().__init__(flags=Qt.WindowFlags(Qt.Window))
        self._sequence = sequence

        self.rows = []  # Initialize rows here
        self.ui = self.init_ui()
        self._connect_buttons()
        self.generate_table()  # Generate the table once during initialization

        self._timer = QTimer()
        self._timer.timeout.connect(self._update_ui)
        self.timer.start(1000)

    def init_ui(self) -> SimpleNamespace:
        # Title
        ui = SimpleNamespace()  # Initialize SimpleNamespace for UI components

        ui.title_label = QLabel(self._sequence.get_label())
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        ui.title_label.setFont(title_font)
        ui.title_label.setAlignment(Qt.AlignCenter)

        # Horizontal line
        ui.hline = QFrame()
        ui.hline.setFrameShape(QFrame.HLine)
        ui.hline.setFrameShadow(QFrame.Sunken)

        # Log section
        ui.log_text_edit = QTextEdit()
        ui.log_text_edit.setReadOnly(True)

        # Buttons
        ui.start_button = QPushButton("Start Sequence")
        ui.stop_button = QPushButton("Stop Sequence")
        ui.taskomat_button = QPushButton("Open Taskomat")
        ui.exit_button = QPushButton("Exit")

        ui.start_button.setEnabled(True)  # Enable the Start button initially
        ui.stop_button.setEnabled(False)  # Initially disable the Stop button

        button_layout = QHBoxLayout()
        button_layout.addWidget(ui.start_button)
        button_layout.addWidget(ui.stop_button)
        button_layout.addWidget(ui.taskomat_button)
        button_layout.addWidget(ui.exit_button)

        # Main layout
        ui.layout = QVBoxLayout()
        ui.layout.addWidget(ui.title_label)
        ui.form_layout = QFormLayout()
        ui.layout.addLayout(ui.form_layout)
        ui.layout.addWidget(ui.hline)
        ui.layout.addWidget(ui.log_text_edit)
        ui.layout.addLayout(button_layout)
        self.setLayout(ui.layout)

        return ui

    def _connect_buttons(self) -> None:
        self.ui.start_button.clicked.connect(self.start_sequence)
        self.ui.stop_button.clicked.connect(self.stop_sequence)
        self.ui.exit_button.clicked.connect(self.close)
        self.ui.taskomat_button.clicked.connect(self.open_taskomat)

    def generate_table(self) -> None:
        # Populate the form layout
        self.rows.clear()  # Clear the rows list
        for irow in range(self._taskomat.get_number_of_steps()):
            hbox = QHBoxLayout()

            number_label = QLabel(str(irow + 1))
            number_label.setFixedWidth(20)  # Set a fixed width for the number label

            step_label = self._taskomat.get_step_label(irow)
            is_disabled = self._taskomat.is_step_disabled(irow)

            description_label = StrikethroughLabel(step_label)
            description_label.set_strikethrough(is_disabled)
            description_label.setWordWrap(True)
            description_label.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Preferred
            )

            status_indicator = CircleIndicator()

            hbox.addWidget(number_label)
            hbox.addWidget(description_label)
            hbox.addWidget(status_indicator, alignment=Qt.AlignRight)

            self.ui.form_layout.addRow(hbox)
            self.rows.append((description_label, status_indicator))

    def set_strikethrough(self, row: int, enabled: bool) -> None:
        self.rows[row][0].set_strikethrough(enabled)

    def set_indicator_colour(self, row: int, colour: QColor) -> None:
        self.rows[row][1].set_colour(colour)

    def _update_ui(self) -> None:
        for irow in range(self._taskomat.get_number_of_steps()):
            is_disabled = self._taskomat.is_step_disabled(irow)
            self.set_strikethrough(irow, is_disabled)
            if self._taskomat.is_step_running(irow):
                # Green
                self.set_indicator_colour(QColor(148, 255, 67))
            elif self._taskomat.is_step_error(irow):
                # Red
                self.set_indicator_colour(QColor(227, 49, 30))
            else:
                # Grey
                self.set_indicator_colour(QColor(63, 63, 63))

        self._update_log()

    def _update_log(self) -> None:
        self.ui.log_text_edit.setHtml(self._taskomat.get_html_log())

    def start_sequence(self) -> None:
        self.ui.start_button.setEnabled(False)
        self.ui.stop_button.setEnabled(True)
        self._sequence.run_once()

    def stop_sequence(self) -> None:
        self.ui.start_button.setEnabled(True)
        self.ui.stop_button.setEnabled(False)
        self.ui._sequence.force_stop()

    def open_taskomat(self) -> None:
        # Some sort of qprocess stuff here, ideally opening to the right panel.
        process = QProcess()
        process.setProgram("jddd-run")
        process.setArguments(
            [
                "-file",
                "taskomat_main.xml",
                "-address",
                f"XFEL.UTIL/TASKOMAT/{self._sequence.location}/",
            ]
        )
        # Take no ownership over the process
        process.startDetached()
