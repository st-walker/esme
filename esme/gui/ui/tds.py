# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'tds.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_tds_control_panel(object):
    def setupUi(self, tds_control_panel):
        tds_control_panel.setObjectName("tds_control_panel")
        tds_control_panel.setEnabled(True)
        tds_control_panel.resize(379, 215)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.MinimumExpanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(tds_control_panel.sizePolicy().hasHeightForWidth())
        tds_control_panel.setSizePolicy(sizePolicy)
        tds_control_panel.setMinimumSize(QtCore.QSize(0, 138))
        tds_control_panel.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.gridLayout = QtWidgets.QGridLayout(tds_control_panel)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.calibration_label = QtWidgets.QLabel(tds_control_panel)
        self.calibration_label.setScaledContents(False)
        self.calibration_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.calibration_label.setObjectName("calibration_label")
        self.horizontalLayout_10.addWidget(self.calibration_label)
        self.calibration_file_path_label = QtWidgets.QLabel(tds_control_panel)
        self.calibration_file_path_label.setTextInteractionFlags(QtCore.Qt.LinksAccessibleByMouse|QtCore.Qt.TextSelectableByMouse)
        self.calibration_file_path_label.setObjectName("calibration_file_path_label")
        self.horizontalLayout_10.addWidget(self.calibration_file_path_label)
        self.gridLayout.addLayout(self.horizontalLayout_10, 5, 0, 1, 2)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.phase_label_2 = QtWidgets.QLabel(tds_control_panel)
        self.phase_label_2.setObjectName("phase_label_2")
        self.horizontalLayout_3.addWidget(self.phase_label_2)
        self.tds_phase_spinbox = QtWidgets.QDoubleSpinBox(tds_control_panel)
        self.tds_phase_spinbox.setMinimum(-1080.0)
        self.tds_phase_spinbox.setMaximum(1080.0)
        self.tds_phase_spinbox.setObjectName("tds_phase_spinbox")
        self.horizontalLayout_3.addWidget(self.tds_phase_spinbox)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.tds_phase_readback_line = QtWidgets.QLineEdit(tds_control_panel)
        self.tds_phase_readback_line.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.tds_phase_readback_line.setMaxLength(4)
        self.tds_phase_readback_line.setReadOnly(True)
        self.tds_phase_readback_line.setClearButtonEnabled(False)
        self.tds_phase_readback_line.setObjectName("tds_phase_readback_line")
        self.horizontalLayout_4.addWidget(self.tds_phase_readback_line)
        self.horizontalLayout_8.addLayout(self.horizontalLayout_4)
        self.gridLayout.addLayout(self.horizontalLayout_8, 0, 0, 1, 2)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.calibration_time_label = QtWidgets.QLabel(tds_control_panel)
        self.calibration_time_label.setObjectName("calibration_time_label")
        self.horizontalLayout_7.addWidget(self.calibration_time_label)
        self.calibration_time_label_2 = QtWidgets.QLabel(tds_control_panel)
        self.calibration_time_label_2.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        self.calibration_time_label_2.setObjectName("calibration_time_label_2")
        self.horizontalLayout_7.addWidget(self.calibration_time_label_2)
        self.gridLayout.addLayout(self.horizontalLayout_7, 6, 0, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.voltage_label = QtWidgets.QLabel(tds_control_panel)
        self.voltage_label.setEnabled(False)
        self.voltage_label.setObjectName("voltage_label")
        self.horizontalLayout.addWidget(self.voltage_label)
        self.tds_voltage_spinbox = QtWidgets.QDoubleSpinBox(tds_control_panel)
        self.tds_voltage_spinbox.setEnabled(False)
        self.tds_voltage_spinbox.setObjectName("tds_voltage_spinbox")
        self.horizontalLayout.addWidget(self.tds_voltage_spinbox)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 0, 1, 1)
        self.tds_calibration_pushbutton = QtWidgets.QPushButton(tds_control_panel)
        self.tds_calibration_pushbutton.setObjectName("tds_calibration_pushbutton")
        self.gridLayout.addWidget(self.tds_calibration_pushbutton, 4, 1, 1, 1)
        self.pushButton = QtWidgets.QPushButton(tds_control_panel)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 4, 0, 1, 1)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.amplitude_label = QtWidgets.QLabel(tds_control_panel)
        self.amplitude_label.setObjectName("amplitude_label")
        self.horizontalLayout_2.addWidget(self.amplitude_label)
        self.tds_amplitude_spinbox = QtWidgets.QDoubleSpinBox(tds_control_panel)
        self.tds_amplitude_spinbox.setMaximum(100.0)
        self.tds_amplitude_spinbox.setObjectName("tds_amplitude_spinbox")
        self.horizontalLayout_2.addWidget(self.tds_amplitude_spinbox)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.tds_amplitude_readback_line = QtWidgets.QLineEdit(tds_control_panel)
        self.tds_amplitude_readback_line.setStyleSheet("background-color: rgba(255, 255, 255, 0);")
        self.tds_amplitude_readback_line.setMaxLength(4)
        self.tds_amplitude_readback_line.setReadOnly(True)
        self.tds_amplitude_readback_line.setObjectName("tds_amplitude_readback_line")
        self.horizontalLayout_5.addWidget(self.tds_amplitude_readback_line)
        self.horizontalLayout_9.addLayout(self.horizontalLayout_5)
        self.gridLayout.addLayout(self.horizontalLayout_9, 1, 0, 1, 2)

        self.retranslateUi(tds_control_panel)
        QtCore.QMetaObject.connectSlotsByName(tds_control_panel)

    def retranslateUi(self, tds_control_panel):
        _translate = QtCore.QCoreApplication.translate
        tds_control_panel.setWindowTitle(_translate("tds_control_panel", "Form"))
        self.calibration_label.setText(_translate("tds_control_panel", "Calibration:"))
        self.calibration_file_path_label.setText(_translate("tds_control_panel", "file"))
        self.phase_label_2.setText(_translate("tds_control_panel", "Phase / °"))
        self.calibration_time_label.setText(_translate("tds_control_panel", "Calibration Time:"))
        self.calibration_time_label_2.setText(_translate("tds_control_panel", "time"))
        self.voltage_label.setText(_translate("tds_control_panel", "Voltage / MV"))
        self.tds_calibration_pushbutton.setText(_translate("tds_control_panel", "Calibration..."))
        self.pushButton.setText(_translate("tds_control_panel", "Find Phase"))
        self.amplitude_label.setText(_translate("tds_control_panel", "Ampl. / %"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tds_control_panel = QtWidgets.QWidget()
    ui = Ui_tds_control_panel()
    ui.setupUi(tds_control_panel)
    tds_control_panel.show()
    sys.exit(app.exec_())
