# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tds_calibration.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_TDSCalibrationWindow(object):
    def setupUi(self, TDSCalibrationWindow):
        TDSCalibrationWindow.setObjectName("TDSCalibrationWindow")
        TDSCalibrationWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(TDSCalibrationWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.widget_plot_1 = MatplotlibCanvas(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_plot_1.sizePolicy().hasHeightForWidth())
        self.widget_plot_1.setSizePolicy(sizePolicy)
        self.widget_plot_1.setObjectName("widget_plot_1")
        self.gridLayout.addWidget(self.widget_plot_1, 0, 0, 2, 1)
        self.calibration_info = QtWidgets.QTextEdit(self.centralwidget)
        self.calibration_info.setObjectName("calibration_info")
        self.gridLayout.addWidget(self.calibration_info, 0, 1, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.apply_calibration_button = QtWidgets.QPushButton(self.centralwidget)
        self.apply_calibration_button.setObjectName("apply_calibration_button")
        self.horizontalLayout.addWidget(self.apply_calibration_button)
        self.update_voltage_button = QtWidgets.QPushButton(self.centralwidget)
        self.update_voltage_button.setObjectName("update_voltage_button")
        self.horizontalLayout.addWidget(self.update_voltage_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.load_tds_calibration_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_tds_calibration_button.setObjectName("load_tds_calibration_button")
        self.horizontalLayout_2.addWidget(self.load_tds_calibration_button)
        self.save_tds_calibration_button = QtWidgets.QPushButton(self.centralwidget)
        self.save_tds_calibration_button.setObjectName("save_tds_calibration_button")
        self.horizontalLayout_2.addWidget(self.save_tds_calibration_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout.addLayout(self.verticalLayout, 1, 1, 1, 1)
        self.widget_plot_2 = MatplotlibCanvas(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_plot_2.sizePolicy().hasHeightForWidth())
        self.widget_plot_2.setSizePolicy(sizePolicy)
        self.widget_plot_2.setObjectName("widget_plot_2")
        self.gridLayout.addWidget(self.widget_plot_2, 2, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.location_label = QtWidgets.QLabel(self.groupBox_2)
        self.location_label.setObjectName("location_label")
        self.horizontalLayout_5.addWidget(self.location_label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.i1d_radio_button = QtWidgets.QRadioButton(self.groupBox_2)
        self.i1d_radio_button.setChecked(True)
        self.i1d_radio_button.setObjectName("i1d_radio_button")
        self.horizontalLayout_4.addWidget(self.i1d_radio_button)
        self.b2d_radio_buton = QtWidgets.QRadioButton(self.groupBox_2)
        self.b2d_radio_buton.setEnabled(False)
        self.b2d_radio_buton.setCheckable(False)
        self.b2d_radio_buton.setObjectName("b2d_radio_buton")
        self.horizontalLayout_4.addWidget(self.b2d_radio_buton)
        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)
        self.gridLayout_2.addLayout(self.horizontalLayout_5, 0, 0, 1, 1)
        self.start_voltage_calibration_button = QtWidgets.QPushButton(self.groupBox_2)
        self.start_voltage_calibration_button.setObjectName("start_voltage_calibration_button")
        self.gridLayout_2.addWidget(self.start_voltage_calibration_button, 1, 0, 1, 1)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.phase_step_label = QtWidgets.QLabel(self.groupBox_2)
        self.phase_step_label.setObjectName("phase_step_label")
        self.horizontalLayout_10.addWidget(self.phase_step_label)
        self.phase_step_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.phase_step_spinner.setObjectName("phase_step_spinner")
        self.horizontalLayout_10.addWidget(self.phase_step_spinner)
        self.gridLayout_2.addLayout(self.horizontalLayout_10, 2, 0, 1, 1)
        self.sample_amplitudes_layout = QtWidgets.QHBoxLayout()
        self.sample_amplitudes_layout.setObjectName("sample_amplitudes_layout")
        self.sample_amplitudes_label = QtWidgets.QLabel(self.groupBox_2)
        self.sample_amplitudes_label.setObjectName("sample_amplitudes_label")
        self.sample_amplitudes_layout.addWidget(self.sample_amplitudes_label)
        self.amplitudes_line_edit = QtWidgets.QLineEdit(self.groupBox_2)
        self.amplitudes_line_edit.setObjectName("amplitudes_line_edit")
        self.sample_amplitudes_layout.addWidget(self.amplitudes_line_edit)
        self.gridLayout_2.addLayout(self.sample_amplitudes_layout, 3, 0, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 2, 1, 1, 1)
        TDSCalibrationWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(TDSCalibrationWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        TDSCalibrationWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(TDSCalibrationWindow)
        self.statusbar.setObjectName("statusbar")
        TDSCalibrationWindow.setStatusBar(self.statusbar)
        self.actionPrint_to_Logbook = QtWidgets.QAction(TDSCalibrationWindow)
        self.actionPrint_to_Logbook.setObjectName("actionPrint_to_Logbook")
        self.actionQuit = QtWidgets.QAction(TDSCalibrationWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuMenu.addAction(self.actionPrint_to_Logbook)
        self.menuMenu.addAction(self.actionQuit)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(TDSCalibrationWindow)
        QtCore.QMetaObject.connectSlotsByName(TDSCalibrationWindow)

    def retranslateUi(self, TDSCalibrationWindow):
        _translate = QtCore.QCoreApplication.translate
        TDSCalibrationWindow.setWindowTitle(_translate("TDSCalibrationWindow", "MainWindow"))
        self.calibration_info.setHtml(_translate("TDSCalibrationWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">percentages = [8, 10, 13, 15, 17, 18, 19, 20, 24]</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">tds_slopes = [204, 245, 320, 388, 464, 492, 548, 589, 670]</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">tds_slope_units = &quot;um/ps&quot;</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">screen_name=&quot;OTRC.64.I1D&quot;</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"># Warning: TDS Calibration <span style=\" font-weight:700;\">MUST</span> be done at OTRC.64.I1D (to be fixed)</p></body></html>"))
        self.apply_calibration_button.setText(_translate("TDSCalibrationWindow", "Apply"))
        self.update_voltage_button.setText(_translate("TDSCalibrationWindow", "Recalculate Voltages"))
        self.load_tds_calibration_button.setText(_translate("TDSCalibrationWindow", "Load"))
        self.save_tds_calibration_button.setText(_translate("TDSCalibrationWindow", "Save "))
        self.groupBox_2.setTitle(_translate("TDSCalibrationWindow", "Time Calibration"))
        self.location_label.setText(_translate("TDSCalibrationWindow", "Location"))
        self.i1d_radio_button.setText(_translate("TDSCalibrationWindow", "I1D"))
        self.b2d_radio_buton.setText(_translate("TDSCalibrationWindow", "B2D"))
        self.start_voltage_calibration_button.setText(_translate("TDSCalibrationWindow", "Start Calibration"))
        self.phase_step_label.setText(_translate("TDSCalibrationWindow", "Phase Step"))
        self.sample_amplitudes_label.setText(_translate("TDSCalibrationWindow", "Sample Amplitudes"))
        self.menuMenu.setTitle(_translate("TDSCalibrationWindow", "Menu"))
        self.actionPrint_to_Logbook.setText(_translate("TDSCalibrationWindow", "Print to Logbook"))
        self.actionQuit.setText(_translate("TDSCalibrationWindow", "Quit"))
from esme.gui.widgets.mpl_widget import MatplotlibCanvas


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    TDSCalibrationWindow = QtWidgets.QMainWindow()
    ui = Ui_TDSCalibrationWindow()
    ui.setupUi(TDSCalibrationWindow)
    TDSCalibrationWindow.show()
    sys.exit(app.exec_())
