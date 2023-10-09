import sys
import logging
import time

import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import QObject, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QApplication, QFileDialog, QFrame, QMainWindow, QMessageBox
import numpy as np


from esme.gui.ui import lps
from esme.control.sbunches import DiagnosticRegion
from esme.control.pattern import get_beam_regions, get_bunch_pattern
from esme.gui.common import build_default_machine_interface, QPlainTextEditLogger, setup_screen_display_widget, send_widget_to_log
from esme.gui.scannerpanel import ProcessedImage, ScanType
from esme.plot import pretty_parameter_table


def start_hires_gui():
    app = QApplication(sys.argv)

    main_window = HighResolutionEnergySpreadMainWindow()

    main_window.show()
    main_window.raise_()
    sys.exit(app.exec_())


class HighResolutionEnergySpreadMainWindow(QMainWindow):
    location = pyqtSignal(object)
    def __init__(self):
        super().__init__()
        self.ui = lps.Ui_hires_mainwindow()
        self.ui.setupUi(self)

        self.setup_logger()

        self.machine = build_default_machine_interface()

        self.scannerp = self.ui.scanner_panel

        self.image_plot = setup_screen_display_widget(self.ui.image_plot)
        self.dispersion_widths_scatter = make_pixel_widths_scatter(self.ui.dispersion_pixel_size_plot_widget,
                                                                   title="Dispersion Scan Peak Energy Slice Widths",
                                                                   xlabel="Dispersion", xunits="m",
                                                                   ylabel="Widths", yunits="px")
        self.voltage_widths_scatter = make_pixel_widths_scatter(self.ui.voltage_pixel_size_plot_widget,
                                                                title="TDS Voltage Scan Peak Energy Slice Widths",
                                                                xlabel="Voltage", xunits="V",
                                                                ylabel="Widths", yunits="px")

        self.beta_widths_scatter = make_pixel_widths_scatter(self.ui.beta_pixel_size_plot,
                                                             title="Beta Scan Peak Energy Slice Widths",
                                                             xlabel="Beta", xunits="m",
                                                             ylabel="Widths", yunits="px")

        
        self.scannerp.processed_image_signal.connect(self.post_processed_image)
        self.scannerp.background_image_signal.connect(self.post_background_image)
        self.scannerp.full_measurement_result_signal.connect(self.post_final_result)

        self.ui.action_print_to_logbook.triggered.connect(self.send_to_logbook)

        self.timer = self.build_main_timer(100)
        self.finished = False

    def send_to_logbook(self):
        send_widget_to_log(self, author="High Res. Energy Spread Measurement")

    def post_background_image(self, image):
        items = self.image_plot.items
        assert len(items) == 1
            
        image_item = items[0]
        image_item.setImage(image)
        

    def post_final_result(self, final_result):
        self.finished = True
        print(final_result)

    def build_main_timer(self, period):
        timer = QTimer()
        timer.timeout.connect(lambda: None)
        # timer.timeout.connect(self.update)
        timer.start(period)
        return timer

    def post_processed_image(self, processed_image: ProcessedImage):
        if self.finished:
            self.dispersion_widths_scatter.setData([], [])
            self.voltage_widths_scatter.setData([], [])
            self.finished = False

        image = processed_image.image
        peak_energy_row = processed_image.central_width_row
        # image[peak_energy_row] *= 0
        items = self.image_plot.items
        assert len(items) == 1
        # view = self.image_plot.items[0].getView()
        # infinite_line = pg.InfiniteLine(pos=peak_energy_row, angle=0)
        # view.addItem(infinite_line)

        if processed_image.scan_type is ScanType.DISPERSION:
            scatter_data = [processed_image.dispersion], [processed_image.central_width.n]
            self.dispersion_widths_scatter.addPoints(*scatter_data)
        elif processed_image.scan_type is ScanType.TDS:
            scatter_data = [processed_image.voltage], [processed_image.central_width.n]
            self.voltage_widths_scatter.addPoints(*scatter_data)
        elif processed_image.scan_type is ScanType.BETA:
            scatter_data = [processed_image.beta], [processed_image.central_width.n]
            self.beta_widths_scatter.addPoints(*scatter_data)
            
        image_item = items[0]
        image_item.setImage(image)

    def setup_logger(self):
        log_handler = QPlainTextEditLogger()
        logging.getLogger().addHandler(log_handler)
        log_handler.log_signal.connect(self.ui.info_log_box.append)

    # def setup_beam_image(self)


        # self.location.connect(self.ui.special_bunch_panel.update_location)
        # self.location.connect(self.ui.tds_panel.update_location)

        # self.connect_buttons()
        # self.setup_indicators()

        # self.image_plot = setup_screen_display_widget(self.ui.screen_display_widget)
        # self.screen_worker, self.screen_thread = self.setup_screen_worker()

        # self.timer = self.build_main_timer(period=100)


def make_pixel_widths_scatter(widget, title, xlabel, xunits, ylabel, yunits):
    plot = widget.addPlot(title=title)

    plot.setLabel('bottom', xlabel, units=xunits)
    plot.setLabel('left', ylabel, units=yunits)
    
    scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen(None), brush=pg.mkBrush(255, 0, 0, 100))
    plot.addItem(scatter)

    return scatter


if __name__ == '__main__':
    main()
