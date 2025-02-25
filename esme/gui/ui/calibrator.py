# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibrator.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_calibrator_mainwindow(object):
    def setupUi(self, calibrator_mainwindow):
        calibrator_mainwindow.setObjectName("calibrator_mainwindow")
        calibrator_mainwindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(calibrator_mainwindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.area_control = AreaControl(self.centralwidget)
        self.area_control.setObjectName("area_control")
        self.gridLayout_4.addWidget(self.area_control, 0, 0, 1, 2)
        self.start_calibration_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_calibration_button.setObjectName("start_calibration_button")
        self.gridLayout_4.addWidget(self.start_calibration_button, 1, 0, 1, 1)
        self.cancel_button = QtWidgets.QPushButton(self.centralwidget)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setObjectName("cancel_button")
        self.gridLayout_4.addWidget(self.cancel_button, 1, 1, 1, 1)
        self.load_calibration_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_calibration_button.setObjectName("load_calibration_button")
        self.gridLayout_4.addWidget(self.load_calibration_button, 2, 0, 1, 1)
        self.check_phases_button = QtWidgets.QPushButton(self.centralwidget)
        self.check_phases_button.setObjectName("check_phases_button")
        self.gridLayout_4.addWidget(self.check_phases_button, 2, 1, 1, 1)
        self.calibration_parameters_group_box = QtWidgets.QGroupBox(self.centralwidget)
        self.calibration_parameters_group_box.setObjectName("calibration_parameters_group_box")
        self.gridLayout_10 = QtWidgets.QGridLayout(self.calibration_parameters_group_box)
        self.gridLayout_10.setObjectName("gridLayout_10")
        self.r12_streaking_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.r12_streaking_label.setObjectName("r12_streaking_label")
        self.gridLayout_10.addWidget(self.r12_streaking_label, 6, 0, 1, 1)
        self.screen_position_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.screen_position_label.setObjectName("screen_position_label")
        self.gridLayout_10.addWidget(self.screen_position_label, 4, 0, 1, 1)
        self.screen_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.screen_label.setObjectName("screen_label")
        self.gridLayout_10.addWidget(self.screen_label, 3, 0, 1, 1)
        self.tds_frequency_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.tds_frequency_label.setObjectName("tds_frequency_label")
        self.gridLayout_10.addWidget(self.tds_frequency_label, 2, 0, 1, 1)
        self.screen_value_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.screen_value_label.setText("")
        self.screen_value_label.setObjectName("screen_value_label")
        self.gridLayout_10.addWidget(self.screen_value_label, 3, 1, 1, 1)
        self.tds_frequency_value_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        font = QtGui.QFont()
        font.setFamily(".AppleSystemUIFont")
        self.tds_frequency_value_label.setFont(font)
        self.tds_frequency_value_label.setObjectName("tds_frequency_value_label")
        self.gridLayout_10.addWidget(self.tds_frequency_value_label, 2, 1, 1, 1)
        self.beam_energy_value_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.beam_energy_value_label.setText("")
        self.beam_energy_value_label.setObjectName("beam_energy_value_label")
        self.gridLayout_10.addWidget(self.beam_energy_value_label, 1, 1, 1, 1)
        self.r12_streaking_value_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.r12_streaking_value_label.setText("")
        self.r12_streaking_value_label.setObjectName("r12_streaking_value_label")
        self.gridLayout_10.addWidget(self.r12_streaking_value_label, 6, 1, 1, 1)
        self.screen_position_value_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.screen_position_value_label.setText("")
        self.screen_position_value_label.setObjectName("screen_position_value_label")
        self.gridLayout_10.addWidget(self.screen_position_value_label, 4, 1, 1, 1)
        self.beam_energy_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.beam_energy_label.setObjectName("beam_energy_label")
        self.gridLayout_10.addWidget(self.beam_energy_label, 1, 0, 1, 1)
        self.pixel_size_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.pixel_size_label.setObjectName("pixel_size_label")
        self.gridLayout_10.addWidget(self.pixel_size_label, 5, 0, 1, 1)
        self.pixel_size_value_label = QtWidgets.QLabel(self.calibration_parameters_group_box)
        self.pixel_size_value_label.setText("")
        self.pixel_size_value_label.setObjectName("pixel_size_value_label")
        self.gridLayout_10.addWidget(self.pixel_size_value_label, 5, 1, 1, 1)
        self.gridLayout_4.addWidget(self.calibration_parameters_group_box, 3, 0, 1, 2)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_4.addWidget(self.line, 4, 0, 1, 2)
        self.table_stack = QtWidgets.QStackedWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.table_stack.sizePolicy().hasHeightForWidth())
        self.table_stack.setSizePolicy(sizePolicy)
        self.table_stack.setObjectName("table_stack")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.gridLayout = QtWidgets.QGridLayout(self.page)
        self.gridLayout.setObjectName("gridLayout")
        self.i1_log = QtWidgets.QTextEdit(self.page)
        self.i1_log.setReadOnly(True)
        self.i1_log.setObjectName("i1_log")
        self.gridLayout.addWidget(self.i1_log, 1, 0, 1, 1)
        self.table_stack.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.page_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.b2_log = QtWidgets.QTextEdit(self.page_2)
        self.b2_log.setReadOnly(True)
        self.b2_log.setObjectName("b2_log")
        self.gridLayout_2.addWidget(self.b2_log, 1, 0, 1, 1)
        self.table_stack.addWidget(self.page_2)
        self.gridLayout_4.addWidget(self.table_stack, 5, 0, 1, 2)
        self.plot_stack = QtWidgets.QStackedWidget(self.centralwidget)
        self.plot_stack.setObjectName("plot_stack")
        self.page_3 = QtWidgets.QWidget()
        self.page_3.setObjectName("page_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.page_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.i1_pscan_graphics = GraphicsLayoutWidget(self.page_3)
        self.i1_pscan_graphics.setObjectName("i1_pscan_graphics")
        self.gridLayout_3.addWidget(self.i1_pscan_graphics, 2, 0, 1, 1)
        self.i1_calibration_graphics = GraphicsLayoutWidget(self.page_3)
        self.i1_calibration_graphics.setObjectName("i1_calibration_graphics")
        self.gridLayout_3.addWidget(self.i1_calibration_graphics, 1, 0, 1, 1)
        self.i1_calibration_table_view = CalibrationTableView(self.page_3)
        self.i1_calibration_table_view.setObjectName("i1_calibration_table_view")
        self.gridLayout_3.addWidget(self.i1_calibration_table_view, 0, 0, 1, 1)
        self.plot_stack.addWidget(self.page_3)
        self.page_4 = QtWidgets.QWidget()
        self.page_4.setObjectName("page_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.page_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.b2_pscan_graphics = GraphicsLayoutWidget(self.page_4)
        self.b2_pscan_graphics.setObjectName("b2_pscan_graphics")
        self.gridLayout_5.addWidget(self.b2_pscan_graphics, 2, 0, 1, 1)
        self.b2_calibration_graphics = GraphicsLayoutWidget(self.page_4)
        self.b2_calibration_graphics.setObjectName("b2_calibration_graphics")
        self.gridLayout_5.addWidget(self.b2_calibration_graphics, 1, 0, 1, 1)
        self.b2_calibration_table_view = CalibrationTableView(self.page_4)
        self.b2_calibration_table_view.setCornerButtonEnabled(True)
        self.b2_calibration_table_view.setObjectName("b2_calibration_table_view")
        self.gridLayout_5.addWidget(self.b2_calibration_table_view, 0, 0, 1, 1)
        self.plot_stack.addWidget(self.page_4)
        self.gridLayout_4.addWidget(self.plot_stack, 0, 2, 6, 1)
        calibrator_mainwindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(calibrator_mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        calibrator_mainwindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(calibrator_mainwindow)
        self.statusbar.setObjectName("statusbar")
        calibrator_mainwindow.setStatusBar(self.statusbar)
        self.actionLoad_Calibration = QtWidgets.QAction(calibrator_mainwindow)
        self.actionLoad_Calibration.setObjectName("actionLoad_Calibration")
        self.actionQuit = QtWidgets.QAction(calibrator_mainwindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuFile.addAction(self.actionLoad_Calibration)
        self.menuFile.addAction(self.actionQuit)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(calibrator_mainwindow)
        self.table_stack.setCurrentIndex(0)
        self.plot_stack.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(calibrator_mainwindow)

    def retranslateUi(self, calibrator_mainwindow):
        _translate = QtCore.QCoreApplication.translate
        calibrator_mainwindow.setWindowTitle(_translate("calibrator_mainwindow", "MainWindow"))
        self.start_calibration_button.setText(_translate("calibrator_mainwindow", "Start Calibration"))
        self.cancel_button.setText(_translate("calibrator_mainwindow", "Cancel Calibration"))
        self.load_calibration_button.setText(_translate("calibrator_mainwindow", "Load Calibration..."))
        self.check_phases_button.setText(_translate("calibrator_mainwindow", "Check Phase Pairs"))
        self.calibration_parameters_group_box.setTitle(_translate("calibrator_mainwindow", "Calibration Parameters"))
        self.r12_streaking_label.setText(_translate("calibrator_mainwindow", "<html><head/><body><p><span style=\" font-style:italic;\">R</span><span style=\" vertical-align:sub;\">34</span>:</p></body></html>"))
        self.screen_position_label.setText(_translate("calibrator_mainwindow", "Screen Position:"))
        self.screen_label.setText(_translate("calibrator_mainwindow", "Screen:"))
        self.tds_frequency_label.setText(_translate("calibrator_mainwindow", "TDS Frequency:"))
        self.tds_frequency_value_label.setText(_translate("calibrator_mainwindow", "3 GHz"))
        self.beam_energy_label.setText(_translate("calibrator_mainwindow", "Beam Energy:"))
        self.pixel_size_label.setText(_translate("calibrator_mainwindow", "Pixel Size:"))
        self.menuFile.setTitle(_translate("calibrator_mainwindow", "File"))
        self.actionLoad_Calibration.setText(_translate("calibrator_mainwindow", "Load Calibration..."))
        self.actionQuit.setText(_translate("calibrator_mainwindow", "Quit"))
        self.actionQuit.setShortcut(_translate("calibrator_mainwindow", "Ctrl+Q"))
from esme.gui.caltable import CalibrationTableView
from esme.gui.widgets.area import AreaControl
from pyqtgraph import GraphicsLayoutWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    calibrator_mainwindow = QtWidgets.QMainWindow()
    ui = Ui_calibrator_mainwindow()
    ui.setupUi(calibrator_mainwindow)
    calibrator_mainwindow.show()
    sys.exit(app.exec_())
