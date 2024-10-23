from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QStackedWidget, QVBoxLayout, QWidget

from esme.core import DiagnosticRegion


class DiagnosticStack(QWidget):
    def __init__(
        self,
        parent=None,
        i1_widget: QWidget | None = None,
        b1_widget: QWidget | None = None,
        b2_widget: QWidget | None = None,
    ):
        super().__init__(parent=parent)
        self._stack = QStackedWidget(self)
        if i1_widget is None:
            i1_widget = QWidget()
        if b1_widget is None:
            b1_widget = QWidget()
        if b2_widget is None:
            b2_widget = QWidget()
        self._stack.addWidget(i1_widget)
        self._stack.addWidget(b1_widget)
        self._stack.addWidget(b2_widget)
        self._regions = [DiagnosticRegion.I1, DiagnosticRegion.B1, DiagnosticRegion.B2]
        self._stack.setCurrentIndex(0)

        layout = QVBoxLayout()
        layout.addWidget(self._stack)
        self.setLayout(layout)

    @pyqtSlot(DiagnosticRegion)
    def set_widget_by_region(self, region: DiagnosticRegion) -> None:
        new_index = self._regions.index(region)
        self._stack.setCurrentIndex(new_index)

    def get_widget_by_region(self, region: DiagnosticRegion) -> QWidget:
        index = self._regions.index(region)
        return self._stack.widget(index)
    
    def set_new_widget_by_region(self, region: DiagnosticRegion, new_widget: QWidget) -> None:
        # Get index of the region
        index = self._regions.index(region)
        old_widget = self._stack.widget(index)
        self._stack.removeWidget(old_widget)
        self._stack.insertWidget(index, new_widget)
        self.update()

    def get_current_widget(self) -> QWidget:
        return self._stack.currentWidget()

