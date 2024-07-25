# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tds.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_tds_control_panel(object):
    def setupUi(self, tds_control_panel):
        tds_control_panel.setObjectName("tds_control_panel")
        tds_control_panel.setEnabled(True)
        tds_control_panel.resize(511, 250)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(tds_control_panel.sizePolicy().hasHeightForWidth())
        tds_control_panel.setSizePolicy(sizePolicy)
        tds_control_panel.setMinimumSize(QtCore.QSize(0, 138))
        tds_control_panel.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.gridLayout_2 = QtWidgets.QGridLayout(tds_control_panel)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.phase_label_2 = QtWidgets.QLabel(tds_control_panel)
        self.phase_label_2.setObjectName("phase_label_2")
        self.gridLayout_2.addWidget(self.phase_label_2, 0, 0, 1, 1)
        self.tds_voltage_spinbox = QtWidgets.QDoubleSpinBox(tds_control_panel)
        self.tds_voltage_spinbox.setEnabled(True)
        self.tds_voltage_spinbox.setLocale(
            QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedKingdom)
        )
        self.tds_voltage_spinbox.setMaximum(99.0)
        self.tds_voltage_spinbox.setSingleStep(0.01)
        self.tds_voltage_spinbox.setObjectName("tds_voltage_spinbox")
        self.gridLayout_2.addWidget(self.tds_voltage_spinbox, 2, 1, 1, 1)
        self.tds_amplitude_spinbox = QtWidgets.QDoubleSpinBox(tds_control_panel)
        self.tds_amplitude_spinbox.setLocale(
            QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)
        )
        self.tds_amplitude_spinbox.setMaximum(100.0)
        self.tds_amplitude_spinbox.setSingleStep(0.01)
        self.tds_amplitude_spinbox.setObjectName("tds_amplitude_spinbox")
        self.gridLayout_2.addWidget(self.tds_amplitude_spinbox, 1, 1, 1, 1)
        self.tds_calibration_pushbutton = QtWidgets.QPushButton(tds_control_panel)
        self.tds_calibration_pushbutton.setObjectName("tds_calibration_pushbutton")
        self.gridLayout_2.addWidget(self.tds_calibration_pushbutton, 6, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(tds_control_panel)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 2, 1, 1)
        self.tds_amplitude_readback = ReadOnlyNumberDisplay(tds_control_panel)
        self.tds_amplitude_readback.setObjectName("tds_amplitude_readback")
        self.gridLayout_2.addWidget(self.tds_amplitude_readback, 1, 3, 1, 1)
        self.label_3 = QtWidgets.QLabel(tds_control_panel)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 2, 2, 1, 1)
        self.voltage_label = QtWidgets.QLabel(tds_control_panel)
        self.voltage_label.setEnabled(True)
        self.voltage_label.setObjectName("voltage_label")
        self.gridLayout_2.addWidget(self.voltage_label, 2, 0, 1, 1)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.calibration_file_path_label = QtWidgets.QLabel(tds_control_panel)
        self.calibration_file_path_label.setText("")
        self.calibration_file_path_label.setTextInteractionFlags(
            QtCore.Qt.LinksAccessibleByMouse | QtCore.Qt.TextSelectableByMouse
        )
        self.calibration_file_path_label.setObjectName("calibration_file_path_label")
        self.horizontalLayout_10.addWidget(self.calibration_file_path_label)
        self.gridLayout_2.addLayout(self.horizontalLayout_10, 6, 1, 1, 2)
        self.amplitude_label = QtWidgets.QLabel(tds_control_panel)
        self.amplitude_label.setObjectName("amplitude_label")
        self.gridLayout_2.addWidget(self.amplitude_label, 1, 0, 1, 1)
        self.tds_phase_spinbox = QtWidgets.QDoubleSpinBox(tds_control_panel)
        self.tds_phase_spinbox.setLocale(
            QtCore.QLocale(QtCore.QLocale.English, QtCore.QLocale.UnitedStates)
        )
        self.tds_phase_spinbox.setMinimum(-1080.0)
        self.tds_phase_spinbox.setMaximum(1080.0)
        self.tds_phase_spinbox.setObjectName("tds_phase_spinbox")
        self.gridLayout_2.addWidget(self.tds_phase_spinbox, 0, 1, 1, 1)
        self.tds_voltage_readback = ReadOnlyNumberDisplay(tds_control_panel)
        self.tds_voltage_readback.setObjectName("tds_voltage_readback")
        self.gridLayout_2.addWidget(self.tds_voltage_readback, 2, 3, 1, 1)
        self.tds_phase_readback = ReadOnlyNumberDisplay(tds_control_panel)
        self.tds_phase_readback.setObjectName("tds_phase_readback")
        self.gridLayout_2.addWidget(self.tds_phase_readback, 0, 3, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.calibration_time_label_2 = QtWidgets.QLabel(tds_control_panel)
        self.calibration_time_label_2.setText("")
        self.calibration_time_label_2.setTextInteractionFlags(
            QtCore.Qt.TextSelectableByMouse
        )
        self.calibration_time_label_2.setObjectName("calibration_time_label_2")
        self.horizontalLayout_7.addWidget(self.calibration_time_label_2)
        self.gridLayout_2.addLayout(self.horizontalLayout_7, 6, 3, 1, 1)
        self.line = QtWidgets.QFrame(tds_control_panel)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 5, 0, 1, 4)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.subtract_180deg_button = QtWidgets.QPushButton(tds_control_panel)
        self.subtract_180deg_button.setObjectName("subtract_180deg_button")
        self.gridLayout.addWidget(self.subtract_180deg_button, 1, 1, 1, 1)
        self.go_to_zero_crossing_button = QtWidgets.QPushButton(tds_control_panel)
        self.go_to_zero_crossing_button.setEnabled(False)
        self.go_to_zero_crossing_button.setObjectName("go_to_zero_crossing_button")
        self.gridLayout.addWidget(self.go_to_zero_crossing_button, 1, 0, 1, 1)
        self.find_zero_crossing_button = QtWidgets.QPushButton(tds_control_panel)
        self.find_zero_crossing_button.setEnabled(False)
        self.find_zero_crossing_button.setFlat(False)
        self.find_zero_crossing_button.setObjectName("find_zero_crossing_button")
        self.gridLayout.addWidget(self.find_zero_crossing_button, 0, 0, 1, 1)
        self.set_zero_crossing_button = QtWidgets.QPushButton(tds_control_panel)
        self.set_zero_crossing_button.setObjectName("set_zero_crossing_button")
        self.gridLayout.addWidget(self.set_zero_crossing_button, 0, 1, 1, 2)
        self.ramp_to_button = QtWidgets.QPushButton(tds_control_panel)
        self.ramp_to_button.setEnabled(False)
        self.ramp_to_button.setObjectName("ramp_to_button")
        self.gridLayout.addWidget(self.ramp_to_button, 1, 3, 1, 1)
        self.zero_crossing_label = QtWidgets.QLabel(tds_control_panel)
        self.zero_crossing_label.setObjectName("zero_crossing_label")
        self.gridLayout.addWidget(self.zero_crossing_label, 0, 3, 1, 1)
        self.add_180_deg_button = QtWidgets.QPushButton(tds_control_panel)
        self.add_180_deg_button.setObjectName("add_180_deg_button")
        self.gridLayout.addWidget(self.add_180_deg_button, 1, 2, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 4, 0, 1, 4)
        self.line_2 = QtWidgets.QFrame(tds_control_panel)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 3, 0, 1, 4)
        self.label = QtWidgets.QLabel(tds_control_panel)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 2, 1, 1)

        self.retranslateUi(tds_control_panel)
        QtCore.QMetaObject.connectSlotsByName(tds_control_panel)

    def retranslateUi(self, tds_control_panel):
        _translate = QtCore.QCoreApplication.translate
        tds_control_panel.setWindowTitle(_translate("tds_control_panel", "Form"))
        self.phase_label_2.setText(_translate("tds_control_panel", "Phase"))
        self.tds_voltage_spinbox.setSuffix(_translate("tds_control_panel", "MV"))
        self.tds_amplitude_spinbox.setSuffix(_translate("tds_control_panel", "%"))
        self.tds_calibration_pushbutton.setText(
            _translate("tds_control_panel", "Calibration...")
        )
        self.label_2.setText(_translate("tds_control_panel", "Readback / %"))
        self.label_3.setText(_translate("tds_control_panel", "Readback / MV"))
        self.voltage_label.setText(_translate("tds_control_panel", "Voltage"))
        self.amplitude_label.setText(_translate("tds_control_panel", "Amplitude"))
        self.tds_phase_spinbox.setSuffix(_translate("tds_control_panel", "°"))
        self.subtract_180deg_button.setText(_translate("tds_control_panel", "-180°"))
        self.go_to_zero_crossing_button.setText(
            _translate("tds_control_panel", "Go to Zero Crossing")
        )
        self.find_zero_crossing_button.setText(
            _translate("tds_control_panel", "Find Zero Crossing")
        )
        self.set_zero_crossing_button.setText(
            _translate("tds_control_panel", "Set Zero Crossing")
        )
        self.ramp_to_button.setText(
            _translate("tds_control_panel", "Ramp from 0% to...")
        )
        self.zero_crossing_label.setText(
            _translate("tds_control_panel", "Zero Crossing: Not Set")
        )
        self.add_180_deg_button.setText(_translate("tds_control_panel", "+180°"))
        self.label.setText(_translate("tds_control_panel", "Readback / °"))


from esme.gui.widgets.core import ReadOnlyNumberDisplay

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    tds_control_panel = QtWidgets.QWidget()
    ui = Ui_tds_control_panel()
    ui.setupUi(tds_control_panel)
    tds_control_panel.show()
    sys.exit(app.exec_())
