import matplotlib

matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=13, height=6, dpi=300, **spkwargs):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi, **spkwargs)
        # self.axes = fig.add_subplot(111)
        super(MatplotlibCanvas, self).__init__(self.fig)
