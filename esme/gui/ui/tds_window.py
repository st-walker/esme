# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tds_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.screen_display_widget = GraphicsLayoutWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.screen_display_widget.sizePolicy().hasHeightForWidth())
        self.screen_display_widget.setSizePolicy(sizePolicy)
        self.screen_display_widget.setObjectName("screen_display_widget")
        self.gridLayout_6.addWidget(self.screen_display_widget, 0, 0, 1, 1)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.horizontalLayout_5.addWidget(self.line)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.log_output_widget = QtWidgets.QTextBrowser(self.centralwidget)
        self.log_output_widget.setObjectName("log_output_widget")
        self.verticalLayout_2.addWidget(self.log_output_widget)
        self.line_2 = QtWidgets.QFrame(self.centralwidget)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.verticalLayout_2.addWidget(self.line_2)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.location_layout = QtWidgets.QHBoxLayout()
        self.location_layout.setObjectName("location_layout")
        self.location_label = QtWidgets.QLabel(self.groupBox)
        self.location_label.setObjectName("location_label")
        self.location_layout.addWidget(self.location_label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.i1d_radio_button = QtWidgets.QRadioButton(self.groupBox)
        self.i1d_radio_button.setChecked(True)
        self.i1d_radio_button.setObjectName("i1d_radio_button")
        self.horizontalLayout_4.addWidget(self.i1d_radio_button)
        self.location_layout.addLayout(self.horizontalLayout_4)
        self.gridLayout_3.addLayout(self.location_layout, 0, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.on_beam_push_button = QtWidgets.QPushButton(self.groupBox)
        self.on_beam_push_button.setCheckable(True)
        self.on_beam_push_button.setAutoExclusive(True)
        self.on_beam_push_button.setObjectName("on_beam_push_button")
        self.buttonGroup = QtWidgets.QButtonGroup(MainWindow)
        self.buttonGroup.setObjectName("buttonGroup")
        self.buttonGroup.addButton(self.on_beam_push_button)
        self.horizontalLayout.addWidget(self.on_beam_push_button)
        self.off_beam_push_button = QtWidgets.QPushButton(self.groupBox)
        self.off_beam_push_button.setCheckable(True)
        self.off_beam_push_button.setAutoExclusive(True)
        self.off_beam_push_button.setObjectName("off_beam_push_button")
        self.buttonGroup.addButton(self.off_beam_push_button)
        self.horizontalLayout.addWidget(self.off_beam_push_button)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.plus_180_phase = QtWidgets.QPushButton(self.groupBox)
        self.plus_180_phase.setObjectName("plus_180_phase")
        self.horizontalLayout_2.addWidget(self.plus_180_phase)
        self.minus_180_phase = QtWidgets.QPushButton(self.groupBox)
        self.minus_180_phase.setObjectName("minus_180_phase")
        self.horizontalLayout_2.addWidget(self.minus_180_phase)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.phase_amp_layout = QtWidgets.QVBoxLayout()
        self.phase_amp_layout.setObjectName("phase_amp_layout")
        self.amplitude_layout = QtWidgets.QHBoxLayout()
        self.amplitude_layout.setObjectName("amplitude_layout")
        self.phase_spin_box = QtWidgets.QLabel(self.groupBox)
        self.phase_spin_box.setObjectName("phase_spin_box")
        self.amplitude_layout.addWidget(self.phase_spin_box)
        self.amplitude_spin_box = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.amplitude_spin_box.setObjectName("amplitude_spin_box")
        self.amplitude_layout.addWidget(self.amplitude_spin_box)
        self.phase_amp_layout.addLayout(self.amplitude_layout)
        self.phase_layout = QtWidgets.QHBoxLayout()
        self.phase_layout.setObjectName("phase_layout")
        self.phase_label = QtWidgets.QLabel(self.groupBox)
        self.phase_label.setObjectName("phase_label")
        self.phase_layout.addWidget(self.phase_label)
        self.phase_spin_box_2 = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.phase_spin_box_2.setObjectName("phase_spin_box_2")
        self.phase_layout.addWidget(self.phase_spin_box_2)
        self.phase_amp_layout.addLayout(self.phase_layout)
        self.gridLayout_2.addLayout(self.phase_amp_layout, 0, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 1, 0, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.start_voltage_calibration_button = QtWidgets.QPushButton(self.groupBox_2)
        self.start_voltage_calibration_button.setObjectName("start_voltage_calibration_button")
        self.gridLayout_4.addWidget(self.start_voltage_calibration_button, 0, 0, 1, 1)
        self.save_voltage_calibration_button = QtWidgets.QPushButton(self.groupBox_2)
        self.save_voltage_calibration_button.setObjectName("save_voltage_calibration_button")
        self.gridLayout_4.addWidget(self.save_voltage_calibration_button, 0, 1, 1, 1)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.phase_step_label = QtWidgets.QLabel(self.groupBox_2)
        self.phase_step_label.setObjectName("phase_step_label")
        self.horizontalLayout_10.addWidget(self.phase_step_label)
        self.phase_step_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.phase_step_spinner.setObjectName("phase_step_spinner")
        self.horizontalLayout_10.addWidget(self.phase_step_spinner)
        self.gridLayout_4.addLayout(self.horizontalLayout_10, 1, 0, 1, 1)
        self.load_voltage_calibration_button = QtWidgets.QPushButton(self.groupBox_2)
        self.load_voltage_calibration_button.setObjectName("load_voltage_calibration_button")
        self.gridLayout_4.addWidget(self.load_voltage_calibration_button, 1, 1, 1, 1)
        self.sample_amplitudes_layout = QtWidgets.QHBoxLayout()
        self.sample_amplitudes_layout.setObjectName("sample_amplitudes_layout")
        self.sample_amplitudes_label = QtWidgets.QLabel(self.groupBox_2)
        self.sample_amplitudes_label.setObjectName("sample_amplitudes_label")
        self.sample_amplitudes_layout.addWidget(self.sample_amplitudes_label)
        self.amplitudes_line_edit = QtWidgets.QLineEdit(self.groupBox_2)
        self.amplitudes_line_edit.setObjectName("amplitudes_line_edit")
        self.sample_amplitudes_layout.addWidget(self.amplitudes_line_edit)
        self.gridLayout_4.addLayout(self.sample_amplitudes_layout, 2, 0, 1, 2)
        self.verticalLayout_2.addWidget(self.groupBox_2)
        self.horizontalLayout_5.addLayout(self.verticalLayout_2)
        self.gridLayout_6.addLayout(self.horizontalLayout_5, 0, 1, 2, 1)
        self.voltage_calibration_plot = PlotWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.voltage_calibration_plot.sizePolicy().hasHeightForWidth())
        self.voltage_calibration_plot.setSizePolicy(sizePolicy)
        self.voltage_calibration_plot.setObjectName("voltage_calibration_plot")
        self.gridLayout_6.addWidget(self.voltage_calibration_plot, 1, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionPrint_to_Logbook = QtWidgets.QAction(MainWindow)
        self.actionPrint_to_Logbook.setObjectName("actionPrint_to_Logbook")
        self.actionQuit = QtWidgets.QAction(MainWindow)
        self.actionQuit.setObjectName("actionQuit")
        self.menuMenu.addAction(self.actionPrint_to_Logbook)
        self.menuMenu.addAction(self.actionQuit)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "Control"))
        self.location_label.setText(_translate("MainWindow", "Location"))
        self.i1d_radio_button.setText(_translate("MainWindow", "I1D"))
        self.on_beam_push_button.setText(_translate("MainWindow", "On Beam"))
        self.off_beam_push_button.setText(_translate("MainWindow", "Off Beam"))
        self.plus_180_phase.setText(_translate("MainWindow", "+180"))
        self.minus_180_phase.setText(_translate("MainWindow", "-180"))
        self.phase_spin_box.setText(_translate("MainWindow", "Amplitude"))
        self.phase_label.setText(_translate("MainWindow", "Phase"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Voltage Calibration"))
        self.start_voltage_calibration_button.setText(_translate("MainWindow", "Start Calibration"))
        self.save_voltage_calibration_button.setText(_translate("MainWindow", "Save Calibration"))
        self.phase_step_label.setText(_translate("MainWindow", "Phase Step"))
        self.load_voltage_calibration_button.setText(_translate("MainWindow", "Load Calibration"))
        self.sample_amplitudes_label.setText(_translate("MainWindow", "Sample Amplitudes"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionPrint_to_Logbook.setText(_translate("MainWindow", "Print to Logbook"))
        self.actionQuit.setText(_translate("MainWindow", "Quit"))
from pyqtgraph import GraphicsLayoutWidget, PlotWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())