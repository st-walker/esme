from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QHeaderView, QTableView


class HTMLHeaderView(QHeaderView):
    def __init__(self, orientation, parent=None):
        super().__init__(orientation, parent)

    def paintSection(self, painter, rect, logicalIndex):
        painter.save()
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        header_text = self.model().headerData(
            logicalIndex, self.orientation(), Qt.DisplayRole
        )
        if isinstance(header_text, str):
            doc = QTextDocument()
            doc.setHtml(header_text)

            if self.orientation() == Qt.Horizontal:
                painter.translate(
                    rect.x(), rect.y() + (rect.height() - doc.size().height()) / 2
                )
            else:
                doc.setTextWidth(
                    rect.width()
                )  # Set text width to the width of the rect for vertical header
                painter.translate(
                    rect.x(), rect.y() + (rect.height() - doc.size().height()) / 2
                )

            doc.drawContents(painter)

        painter.restore()


class BeamCurrentTableView(QTableView):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = BeamParametersTableModel()
        self.setModel(self.model)

        # Make columns stretch to fit the available space
        header = self.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.Stretch)

        # Use HTMLHeaderView for row headers
        self.setVerticalHeader(HTMLHeaderView(Qt.Vertical))

        # Measure the width of the rendered HTML text for the vertical header
        max_width = 0
        for header in self.model.html_headers:
            doc = QTextDocument()
            doc.setHtml(header)
            width = doc.size().width()
            if width > max_width:
                max_width = width

        # Set the width of the vertical header sections
        self.verticalHeader().setFixedWidth(int(max_width + 10))  # Add some padding
