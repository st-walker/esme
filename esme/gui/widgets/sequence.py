from types import SimpleNamespace

from PyQt5.QtCore import QProcess, Qt, QTimer
from PyQt5.QtGui import QColor, QFont, QPainter, QPaintEvent, QPen, QTextCursor
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
from esme.gui.widgets.status import CircleIndicator

from esme.control.taskomat import Sequence

class StepLabel(QLabel):
    def __init__(self, keyword: str = "", text: str = "", indentation_level: int = 0, parent=None):
        super().__init__(text, parent)
        self.keyword = keyword
        self.strikethrough = False
        self.indentation_level = indentation_level
        self.setText(self.format_text(keyword, text))

    def format_text(self, keyword: str, text: str) -> str:
        indent = "    " * self.indentation_level
        text = f"{indent}{text}"
        if keyword:
            return f"<span style='color: royalblue; font-weight: bold;'>{keyword}</span> {text}"
        return text
    
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
        self._timer.start(50)

        self._last_log_message: str = ""

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
        nesting_level = 0
        for istep in self._sequence.get_step_numbers():
            hbox = QHBoxLayout()

            number_label = QLabel(str(istep))
            number_label.setFixedWidth(20)  # Set a fixed width for the number label

            step_label = self._sequence.get_step_label(istep)
            is_disabled = self._sequence.is_step_disabled(istep)
            keyword = self._sequence.get_step_type(istep)

            if keyword == "ACTION":
                keyword = ""

            description_label = StepLabel(keyword=keyword, text=step_label, indentation_level=nesting_level)
            description_label.set_strikethrough(is_disabled)
            description_label.setWordWrap(True)
            description_label.setSizePolicy(
                QSizePolicy.Expanding, QSizePolicy.Preferred
            )
            if keyword in {"IF", "WHILE", "TRY", "CATCH"}:
                nesting_level += 1
            elif keyword in {"END"}:
                nesting_level = max(0, nesting_level - 1)

            status_indicator = CircleIndicator()

            hbox.addWidget(number_label)
            hbox.addWidget(description_label)
            hbox.addWidget(status_indicator, alignment=Qt.AlignRight)

            self.ui.form_layout.addRow(hbox)
            self.rows.append((description_label, status_indicator))

    def set_strikethrough(self, row: int, strike: bool) -> None:
        # Counts from zero, so step number 1 has row of 0
        self.rows[row][0].set_strikethrough(strike)

    def set_indicator_colour(self, row: int, colour: QColor) -> None:
        # Counts from zero, so step number 1 has row of 0
        self.rows[row][1].set_colour(colour)

    def _update_ui(self) -> None:
        for step_number in self._sequence.get_step_numbers():
            irow = step_number - 1
            self.set_strikethrough(irow, self._sequence.is_step_disabled(step_number))

            if self._sequence.is_step_running(step_number):
                # Green
                self.set_indicator_colour(irow, QColor(148, 255, 67))
            elif self._sequence.is_step_error(step_number):
                # Red
                self.set_indicator_colour(irow, QColor(227, 49, 30))
            else:
                # Grey
                self.set_indicator_colour(irow, QColor(63, 63, 63))
        self._set_buttons_for_running_state(is_running=self._sequence.is_running())
        self._update_log()

    def _update_log(self) -> None:
        log_text = self._sequence.get_html_log()
        if self._last_log_message == log_text:
            return

        scrollbar = self.ui.log_text_edit.verticalScrollBar()
        old_scroll_value = scrollbar.value()
        at_bottom_of_text = old_scroll_value == scrollbar.maximum() 

        # Now update the contents of the text
        self._last_log_message = log_text
        self.ui.log_text_edit.setHtml(log_text)

        if at_bottom_of_text:
            new_scroll = scrollbar.maximum()
        else:
            new_scroll = old_scroll_value

        scrollbar.setValue(new_scroll)

    def _set_buttons_for_running_state(self, *, is_running) -> None:
        self.ui.start_button.setEnabled(not is_running)
        self.ui.stop_button.setEnabled(is_running)

    def start_sequence(self) -> None:
        self._set_buttons_for_running_state(is_running=True)
        self._sequence.run_once()

    def stop_sequence(self) -> None:
        self._set_buttons_for_running_state(is_running=False)
        self._sequence.force_stop()

    def open_taskomat(self) -> None:
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
