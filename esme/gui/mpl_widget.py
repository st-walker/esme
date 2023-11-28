import sys

import matplotlib

matplotlib.use('Qt5Agg')

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

from PyQt5 import QtCore, QtWidgets


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=5, height=4, dpi=200, **spkwargs):

        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi, **spkwargs)
        # self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
