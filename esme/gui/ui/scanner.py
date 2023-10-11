# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scanner.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_scanner_form(object):
    def setupUi(self, scanner_form):
        scanner_form.setObjectName("scanner_form")
        scanner_form.resize(535, 406)
        self.gridLayout = QtWidgets.QGridLayout(scanner_form)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.do_beta_scan_checkbox = QtWidgets.QCheckBox(scanner_form)
        self.do_beta_scan_checkbox.setChecked(True)
        self.do_beta_scan_checkbox.setObjectName("do_beta_scan_checkbox")
        self.horizontalLayout_12.addWidget(self.do_beta_scan_checkbox)
        self.remeasure_dispersoin_checkbox = QtWidgets.QCheckBox(scanner_form)
        self.remeasure_dispersoin_checkbox.setEnabled(False)
        self.remeasure_dispersoin_checkbox.setObjectName("remeasure_dispersoin_checkbox")
        self.horizontalLayout_12.addWidget(self.remeasure_dispersoin_checkbox)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.label_3 = QtWidgets.QLabel(scanner_form)
        self.label_3.setTextFormat(QtCore.Qt.RichText)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_9.addWidget(self.label_3)
        self.measured_emittance_spinbox = QtWidgets.QDoubleSpinBox(scanner_form)
        self.measured_emittance_spinbox.setDecimals(3)
        self.measured_emittance_spinbox.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.measured_emittance_spinbox.setObjectName("measured_emittance_spinbox")
        self.horizontalLayout_9.addWidget(self.measured_emittance_spinbox)
        self.horizontalLayout_12.addLayout(self.horizontalLayout_9)
        self.gridLayout.addLayout(self.horizontalLayout_12, 6, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.label = QtWidgets.QLabel(scanner_form)
        self.label.setObjectName("label")
        self.horizontalLayout_8.addWidget(self.label)
        self.dispersion_scan_tds_voltage_spinbox = QtWidgets.QDoubleSpinBox(scanner_form)
        self.dispersion_scan_tds_voltage_spinbox.setMaximum(5.0)
        self.dispersion_scan_tds_voltage_spinbox.setSingleStep(0.1)
        self.dispersion_scan_tds_voltage_spinbox.setObjectName("dispersion_scan_tds_voltage_spinbox")
        self.horizontalLayout_8.addWidget(self.dispersion_scan_tds_voltage_spinbox)
        self.gridLayout.addLayout(self.horizontalLayout_8, 4, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.start_measurement_button = QtWidgets.QPushButton(scanner_form)
        self.start_measurement_button.setCheckable(False)
        self.start_measurement_button.setAutoDefault(False)
        self.start_measurement_button.setDefault(False)
        self.start_measurement_button.setFlat(False)
        self.start_measurement_button.setObjectName("start_measurement_button")
        self.horizontalLayout_3.addWidget(self.start_measurement_button)
        self.stop_measurement_button = QtWidgets.QPushButton(scanner_form)
        self.stop_measurement_button.setObjectName("stop_measurement_button")
        self.horizontalLayout_3.addWidget(self.stop_measurement_button)
        self.gridLayout.addLayout(self.horizontalLayout_3, 8, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.measurement_name_label = QtWidgets.QLabel(scanner_form)
        self.measurement_name_label.setObjectName("measurement_name_label")
        self.horizontalLayout_6.addWidget(self.measurement_name_label)
        self.slug_line_edit = QtWidgets.QLineEdit(scanner_form)
        self.slug_line_edit.setText("")
        self.slug_line_edit.setClearButtonEnabled(False)
        self.slug_line_edit.setObjectName("slug_line_edit")
        self.horizontalLayout_6.addWidget(self.slug_line_edit)
        self.gridLayout.addLayout(self.horizontalLayout_6, 7, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(scanner_form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.tds_voltages = QtWidgets.QLineEdit(scanner_form)
        self.tds_voltages.setText("")
        self.tds_voltages.setObjectName("tds_voltages")
        self.horizontalLayout.addWidget(self.tds_voltages)
        self.gridLayout.addLayout(self.horizontalLayout, 3, 0, 1, 1)
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.background_shots_label = QtWidgets.QLabel(scanner_form)
        self.background_shots_label.setObjectName("background_shots_label")
        self.horizontalLayout_4.addWidget(self.background_shots_label)
        self.bg_shots_spinner = QtWidgets.QSpinBox(scanner_form)
        self.bg_shots_spinner.setProperty("value", 0)
        self.bg_shots_spinner.setObjectName("bg_shots_spinner")
        self.horizontalLayout_4.addWidget(self.bg_shots_spinner)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.beam_shots_label = QtWidgets.QLabel(scanner_form)
        self.beam_shots_label.setObjectName("beam_shots_label")
        self.horizontalLayout_5.addWidget(self.beam_shots_label)
        self.beam_shots_spinner = QtWidgets.QSpinBox(scanner_form)
        self.beam_shots_spinner.setProperty("value", 0)
        self.beam_shots_spinner.setObjectName("beam_shots_spinner")
        self.horizontalLayout_5.addWidget(self.beam_shots_spinner)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        self.gridLayout.addLayout(self.horizontalLayout_7, 1, 0, 1, 1)
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_4 = QtWidgets.QLabel(scanner_form)
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_2.addWidget(self.label_4)
        self.dispersion_setpoint_combo_box = QtWidgets.QComboBox(scanner_form)
        self.dispersion_setpoint_combo_box.setObjectName("dispersion_setpoint_combo_box")
        self.horizontalLayout_2.addWidget(self.dispersion_setpoint_combo_box)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.label_5 = QtWidgets.QLabel(scanner_form)
        self.label_5.setEnabled(True)
        self.label_5.setObjectName("label_5")
        self.horizontalLayout_10.addWidget(self.label_5)
        self.beta_setpoint_combo_box = QtWidgets.QComboBox(scanner_form)
        self.beta_setpoint_combo_box.setEnabled(True)
        self.beta_setpoint_combo_box.setObjectName("beta_setpoint_combo_box")
        self.horizontalLayout_10.addWidget(self.beta_setpoint_combo_box)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_10)
        self.apply_optics_button = QtWidgets.QPushButton(scanner_form)
        self.apply_optics_button.setObjectName("apply_optics_button")
        self.horizontalLayout_11.addWidget(self.apply_optics_button)
        self.gridLayout.addLayout(self.horizontalLayout_11, 2, 0, 1, 1)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.cycle_quads_button = QtWidgets.QPushButton(scanner_form)
        self.cycle_quads_button.setEnabled(True)
        self.cycle_quads_button.setObjectName("cycle_quads_button")
        self.horizontalLayout_13.addWidget(self.cycle_quads_button)
        self.open_jddd_screen_gui_button = QtWidgets.QPushButton(scanner_form)
        self.open_jddd_screen_gui_button.setObjectName("open_jddd_screen_gui_button")
        self.horizontalLayout_13.addWidget(self.open_jddd_screen_gui_button)
        self.preferences_button = QtWidgets.QPushButton(scanner_form)
        self.preferences_button.setObjectName("preferences_button")
        self.horizontalLayout_13.addWidget(self.preferences_button)
        self.gridLayout.addLayout(self.horizontalLayout_13, 5, 0, 1, 1)

        self.retranslateUi(scanner_form)
        QtCore.QMetaObject.connectSlotsByName(scanner_form)

    def retranslateUi(self, scanner_form):
        _translate = QtCore.QCoreApplication.translate
        scanner_form.setWindowTitle(_translate("scanner_form", "Form"))
        self.do_beta_scan_checkbox.setText(_translate("scanner_form", "Do Beta Scan"))
        self.remeasure_dispersoin_checkbox.setText(_translate("scanner_form", "Remeasure Dispersion"))
        self.label_3.setText(_translate("scanner_form", "<html><head/><body><p><span style=\" font-style:italic;\">ε</span><span style=\" vertical-align:sub;\">n </span>/ mm·mrad</p></body></html>"))
        self.label.setText(_translate("scanner_form", "Dispersion Scan TDS Voltage / MV"))
        self.start_measurement_button.setText(_translate("scanner_form", "Start Measurement"))
        self.stop_measurement_button.setText(_translate("scanner_form", "Cancel"))
        self.measurement_name_label.setText(_translate("scanner_form", "Measurement Slug"))
        self.label_2.setText(_translate("scanner_form", "TDS Scan Voltages / MV"))
        self.background_shots_label.setText(_translate("scanner_form", "Background shots "))
        self.beam_shots_label.setText(_translate("scanner_form", "Beam Shots"))
        self.label_4.setText(_translate("scanner_form", "<html><head/><body><p>Screen <span style=\" font-style:italic;\">D</span><span style=\" font-style:italic; vertical-align:sub;\">x </span>/ m</p></body></html>"))
        self.label_5.setText(_translate("scanner_form", "<html><head/><body><p>Screen <span style=\" font-style:italic;\">β</span><span style=\" font-style:italic; vertical-align:sub;\">x </span>/ m</p></body></html>"))
        self.apply_optics_button.setText(_translate("scanner_form", "Apply"))
        self.cycle_quads_button.setText(_translate("scanner_form", "Cycle Quadrupoles"))
        self.open_jddd_screen_gui_button.setText(_translate("scanner_form", "Open Camera Control..."))
        self.preferences_button.setText(_translate("scanner_form", "Configure..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    scanner_form = QtWidgets.QWidget()
    ui = Ui_scanner_form()
    ui.setupUi(scanner_form)
    scanner_form.show()
    sys.exit(app.exec_())
