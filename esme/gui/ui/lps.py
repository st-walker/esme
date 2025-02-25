# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'lps.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_hires_mainwindow(object):
    def setupUi(self, hires_mainwindow):
        hires_mainwindow.setObjectName("hires_mainwindow")
        hires_mainwindow.resize(800, 785)
        self.centralwidget = QtWidgets.QWidget(hires_mainwindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.dispersion_pixel_size_plot_widget = GraphicsLayoutWidget(self.centralwidget)
        self.dispersion_pixel_size_plot_widget.setMinimumSize(QtCore.QSize(289, 170))
        self.dispersion_pixel_size_plot_widget.setObjectName("dispersion_pixel_size_plot_widget")
        self.verticalLayout_2.addWidget(self.dispersion_pixel_size_plot_widget)
        self.voltage_pixel_size_plot_widget = GraphicsLayoutWidget(self.centralwidget)
        self.voltage_pixel_size_plot_widget.setMinimumSize(QtCore.QSize(289, 169))
        self.voltage_pixel_size_plot_widget.setObjectName("voltage_pixel_size_plot_widget")
        self.verticalLayout_2.addWidget(self.voltage_pixel_size_plot_widget)
        self.beta_pixel_size_plot = GraphicsLayoutWidget(self.centralwidget)
        self.beta_pixel_size_plot.setMinimumSize(QtCore.QSize(289, 169))
        self.beta_pixel_size_plot.setObjectName("beta_pixel_size_plot")
        self.verticalLayout_2.addWidget(self.beta_pixel_size_plot)
        self.image_plot = GraphicsLayoutWidget(self.centralwidget)
        self.image_plot.setMinimumSize(QtCore.QSize(289, 170))
        self.image_plot.setObjectName("image_plot")
        self.verticalLayout_2.addWidget(self.image_plot)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 4, 1)
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 1, 1, 1)
        self.info_log_box = QtWidgets.QTextBrowser(self.centralwidget)
        self.info_log_box.setMinimumSize(QtCore.QSize(256, 0))
        self.info_log_box.setMaximumSize(QtCore.QSize(500, 16777215))
        self.info_log_box.setObjectName("info_log_box")
        self.gridLayout.addWidget(self.info_log_box, 1, 1, 1, 2)
        self.scanner_panel = ScannerControl(self.centralwidget)
        self.scanner_panel.setMinimumSize(QtCore.QSize(0, 0))
        self.scanner_panel.setMaximumSize(QtCore.QSize(500, 16777215))
        self.scanner_panel.setObjectName("scanner_panel")
        self.gridLayout.addWidget(self.scanner_panel, 3, 1, 1, 2)
        self.tds_group_box = QtWidgets.QGroupBox(self.centralwidget)
        self.tds_group_box.setMinimumSize(QtCore.QSize(251, 0))
        self.tds_group_box.setMaximumSize(QtCore.QSize(500, 16777215))
        self.tds_group_box.setObjectName("tds_group_box")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tds_group_box)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.tds_panel = TDSControl(self.tds_group_box)
        self.tds_panel.setObjectName("tds_panel")
        self.gridLayout_2.addWidget(self.tds_panel, 0, 0, 1, 1)
        self.gridLayout.addWidget(self.tds_group_box, 2, 1, 1, 2)
        hires_mainwindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(hires_mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setDefaultUp(False)
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuHelp = QtWidgets.QMenu(self.menubar)
        self.menuHelp.setObjectName("menuHelp")
        hires_mainwindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(hires_mainwindow)
        self.statusbar.setObjectName("statusbar")
        hires_mainwindow.setStatusBar(self.statusbar)
        self.action_print_to_logbook = QtWidgets.QAction(hires_mainwindow)
        self.action_print_to_logbook.setObjectName("action_print_to_logbook")
        self.actionMachine_setup = QtWidgets.QAction(hires_mainwindow)
        self.actionMachine_setup.setObjectName("actionMachine_setup")
        self.actionAbout = QtWidgets.QAction(hires_mainwindow)
        self.actionAbout.setObjectName("actionAbout")
        self.actionAbout_HIREIS = QtWidgets.QAction(hires_mainwindow)
        self.actionAbout_HIREIS.setObjectName("actionAbout_HIREIS")
        self.menuMenu.addAction(self.action_print_to_logbook)
        self.menuMenu.addAction(self.actionAbout_HIREIS)
        self.menuHelp.addAction(self.actionMachine_setup)
        self.menuHelp.addAction(self.actionAbout)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuHelp.menuAction())

        self.retranslateUi(hires_mainwindow)
        QtCore.QMetaObject.connectSlotsByName(hires_mainwindow)

    def retranslateUi(self, hires_mainwindow):
        _translate = QtCore.QCoreApplication.translate
        hires_mainwindow.setWindowTitle(_translate("hires_mainwindow", "High Resolution Slice Energy Spread Measurer"))
        self.label.setText(_translate("hires_mainwindow", "Log"))
        self.tds_group_box.setTitle(_translate("hires_mainwindow", "TDS"))
        self.menuMenu.setTitle(_translate("hires_mainwindow", "Menu"))
        self.menuHelp.setTitle(_translate("hires_mainwindow", "Help"))
        self.action_print_to_logbook.setText(_translate("hires_mainwindow", "Print to Logbook"))
        self.action_print_to_logbook.setShortcut(_translate("hires_mainwindow", "Ctrl+P"))
        self.actionMachine_setup.setText(_translate("hires_mainwindow", "Machine Setup"))
        self.actionAbout.setText(_translate("hires_mainwindow", "About"))
        self.actionAbout_HIREIS.setText(_translate("hires_mainwindow", "About HIREIS"))
from esme.gui.widgets.scannerpanel import ScannerControl
from esme.gui.widgets.tds import TDSControl
from pyqtgraph import GraphicsLayoutWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    hires_mainwindow = QtWidgets.QMainWindow()
    ui = Ui_hires_mainwindow()
    ui.setupUi(hires_mainwindow)
    hires_mainwindow.show()
    sys.exit(app.exec_())
