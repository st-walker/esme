from PyQt5.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtWidgets import QHeaderView, QTableView
from PyQt5.QtGui import QTextDocument, QPainter

from dataclasses import dataclass
from scipy.constants import c
import pandas as pd

@dataclass
class MeasuredBeamParameters:
    sigma_t: tuple[float, float] | None = None
    resolution_t: tuple[float, float] | None = None
    sigma_x0: tuple[float, float] | None = None
    sigma_xi: tuple[float, float] | None = None

    @property
    def resolution_z(self):
        return (self.resolution_t[0] * c, self.resolution_t[1] * c)

    @property
    def sigma_z(self):
        return (self.sigma_t[0] * c, self.sigma_z[1] * c)


def format_value_with_uncertainty(value, uncertainty):
    """Format the value with its uncertainty."""
    return f"{value} ± {uncertainty}"


class BeamCurrentTableModel(QAbstractTableModel):
    def __init__(
        self,
        gaussian_params: MeasuredBeamParameters | None = None,
        rms_params: MeasuredBeamParameters | None = None,
    ):
        super().__init__()

        self.gaussian_params = gaussian_params or MeasuredBeamParameters()
        self.rms_params = rms_params or MeasuredBeamParameters()

        self.headers = ["Gaussian", "RMS", "Units"]
        self.param_order = [
            "sigma_t",
            "sigma_z",
            "resolution_t",
            "resolution_z",
            "sigma_x0",
            "sigma_xi",
        ]
        self.units = {
            "sigma_t": "s",
            "sigma_z": "mm",
            "resolution_t": "s",
            "resolution_z": "mm",
            "sigma_x0": "mm",
            "sigma_xi": "mm",
        }
        self.html_headers = [
            "<i>σ<sub>t</sub></i>",
            "<i>σ<sub>z</sub></i>",
            "<i>R<sub>t</sub></i>",
            "<i>R<sub>z</sub></i>",
            "<i>σ<sub>x,0</sub></i>",
            "<i>σ<sub>x,i</sub></i>",
        ]

    def rowCount(self, parent=None):
        return len(self.param_order)

    def columnCount(self, parent=None):
        return 3

    def data(self, index: QModelIndex, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        if role == Qt.DisplayRole:
            row = index.row()
            col = index.column()
            param_name = self.param_order[row]
            try:
                gaussian_value = getattr(self.gaussian_params, param_name)
            except TypeError:
                gaussian_value = None

            try:
                rms_value = getattr(self.rms_params, param_name)
            except TypeError:
                rms_value = None

            if col == 0:
                if gaussian_value is None:
                    return None
                return format_value_with_uncertainty(
                    gaussian_value[0], gaussian_value[1]
                )
            elif col == 1:
                if rms_value is None:
                    return None
                return format_value_with_uncertainty(rms_value[0], rms_value[1])
            elif col == 2:
                return self.units[param_name]
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole) -> str | None:
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return self.headers[section]
            elif orientation == Qt.Vertical:
                return self.html_headers[section]
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the table model to a pandas DataFrame."""
        data = {
            "Parameter": [],
            "Gaussian Value": [],
            "Gaussian Uncertainty": [],
            "RMS Value": [],
            "RMS Uncertainty": [],
            "Units": [],
        }

        for row in range(self.rowCount()):
            param_name = self.param_order[row]
            gaussian_value = getattr(self.gaussian_params, param_name)
            rms_value = getattr(self.rms_params, param_name)

            data["Parameter"].append(param_name)
            data["Gaussian Value"].append(gaussian_value[0])
            data["Gaussian Uncertainty"].append(gaussian_value[1])
            data["RMS Value"].append(rms_value[0])
            data["RMS Uncertainty"].append(rms_value[1])
            data["Units"].append(self.units[param_name])

        return pd.DataFrame(data).set_index("Parameter")

    def to_logbook_printable_table(self):
        return _format_df_for_printing(
            self.to_dataframe(),
            [
                ("Gaussian Value", "Gaussian Uncertainty"),
                ("RMS Value", "RMS Uncertainty"),
            ],
            self.units,
        )

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

        self.model = BeamCurrentTableModel()
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
