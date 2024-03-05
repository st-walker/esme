# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scanner.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_scanner_form(object):
    def setupUi(self, scanner_form):
        scanner_form.setObjectName("scanner_form")
        scanner_form.resize(500, 517)
        scanner_form.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.gridLayout_4 = QtWidgets.QGridLayout(scanner_form)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.groupBox_3 = QtWidgets.QGroupBox(scanner_form)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pick_slice_based_on_position_checkbox = QtWidgets.QCheckBox(self.groupBox_3)
        self.pick_slice_based_on_position_checkbox.setObjectName("pick_slice_based_on_position_checkbox")
        self.gridLayout_3.addWidget(self.pick_slice_based_on_position_checkbox, 3, 0, 1, 1)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.do_beta_scan_checkbox = QtWidgets.QCheckBox(self.groupBox_3)
        self.do_beta_scan_checkbox.setChecked(True)
        self.do_beta_scan_checkbox.setObjectName("do_beta_scan_checkbox")
        self.horizontalLayout_12.addWidget(self.do_beta_scan_checkbox)
        self.remeasure_dispersoin_checkbox = QtWidgets.QCheckBox(self.groupBox_3)
        self.remeasure_dispersoin_checkbox.setEnabled(False)
        self.remeasure_dispersoin_checkbox.setObjectName("remeasure_dispersoin_checkbox")
        self.horizontalLayout_12.addWidget(self.remeasure_dispersoin_checkbox)
        self.do_full_phase_space_checkbox = QtWidgets.QCheckBox(self.groupBox_3)
        self.do_full_phase_space_checkbox.setObjectName("do_full_phase_space_checkbox")
        self.horizontalLayout_12.addWidget(self.do_full_phase_space_checkbox)
        self.gridLayout_3.addLayout(self.horizontalLayout_12, 2, 0, 1, 1)
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.dscan_voltage_label = QtWidgets.QLabel(self.groupBox_3)
        self.dscan_voltage_label.setObjectName("dscan_voltage_label")
        self.horizontalLayout_8.addWidget(self.dscan_voltage_label)
        self.dispersion_scan_tds_voltage_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.dispersion_scan_tds_voltage_spinbox.setMaximum(5.0)
        self.dispersion_scan_tds_voltage_spinbox.setSingleStep(0.1)
        self.dispersion_scan_tds_voltage_spinbox.setObjectName("dispersion_scan_tds_voltage_spinbox")
        self.horizontalLayout_8.addWidget(self.dispersion_scan_tds_voltage_spinbox)
        self.gridLayout_3.addLayout(self.horizontalLayout_8, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.tds_voltages_label = QtWidgets.QLabel(self.groupBox_3)
        self.tds_voltages_label.setObjectName("tds_voltages_label")
        self.horizontalLayout.addWidget(self.tds_voltages_label)
        self.tds_voltages = QtWidgets.QLineEdit(self.groupBox_3)
        self.tds_voltages.setText("")
        self.tds_voltages.setObjectName("tds_voltages")
        self.horizontalLayout.addWidget(self.tds_voltages)
        self.gridLayout_3.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setObjectName("label")
        self.horizontalLayout_15.addWidget(self.label)
        self.slice_selection_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.slice_selection_spinner.setDecimals(2)
        self.slice_selection_spinner.setMinimum(-0.5)
        self.slice_selection_spinner.setMaximum(0.5)
        self.slice_selection_spinner.setSingleStep(0.1)
        self.slice_selection_spinner.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
        self.slice_selection_spinner.setObjectName("slice_selection_spinner")
        self.horizontalLayout_15.addWidget(self.slice_selection_spinner)
        self.gridLayout_3.addLayout(self.horizontalLayout_15, 4, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_3, 1, 0, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(scanner_form)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.background_shots_label = QtWidgets.QLabel(self.groupBox_2)
        self.background_shots_label.setObjectName("background_shots_label")
        self.horizontalLayout_4.addWidget(self.background_shots_label)
        self.bg_shots_spinner = QtWidgets.QSpinBox(self.groupBox_2)
        self.bg_shots_spinner.setProperty("value", 0)
        self.bg_shots_spinner.setObjectName("bg_shots_spinner")
        self.horizontalLayout_4.addWidget(self.bg_shots_spinner)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.beam_shots_label = QtWidgets.QLabel(self.groupBox_2)
        self.beam_shots_label.setObjectName("beam_shots_label")
        self.horizontalLayout_5.addWidget(self.beam_shots_label)
        self.beam_shots_spinner = QtWidgets.QSpinBox(self.groupBox_2)
        self.beam_shots_spinner.setProperty("value", 0)
        self.beam_shots_spinner.setObjectName("beam_shots_spinner")
        self.horizontalLayout_5.addWidget(self.beam_shots_spinner)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        self.gridLayout_2.addLayout(self.horizontalLayout_7, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.start_measurement_button = QtWidgets.QPushButton(self.groupBox_2)
        self.start_measurement_button.setCheckable(False)
        self.start_measurement_button.setAutoDefault(False)
        self.start_measurement_button.setDefault(False)
        self.start_measurement_button.setFlat(False)
        self.start_measurement_button.setObjectName("start_measurement_button")
        self.horizontalLayout_3.addWidget(self.start_measurement_button)
        self.stop_measurement_button = QtWidgets.QPushButton(self.groupBox_2)
        self.stop_measurement_button.setObjectName("stop_measurement_button")
        self.horizontalLayout_3.addWidget(self.stop_measurement_button)
        self.preferences_button = QtWidgets.QPushButton(self.groupBox_2)
        self.preferences_button.setObjectName("preferences_button")
        self.horizontalLayout_3.addWidget(self.preferences_button)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 2, 0, 1, 1)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.measurement_name_label = QtWidgets.QLabel(self.groupBox_2)
        self.measurement_name_label.setObjectName("measurement_name_label")
        self.horizontalLayout_6.addWidget(self.measurement_name_label)
        self.slug_line_edit = QtWidgets.QLineEdit(self.groupBox_2)
        self.slug_line_edit.setEnabled(True)
        self.slug_line_edit.setText("")
        self.slug_line_edit.setClearButtonEnabled(False)
        self.slug_line_edit.setObjectName("slug_line_edit")
        self.horizontalLayout_6.addWidget(self.slug_line_edit)
        self.gridLayout_2.addLayout(self.horizontalLayout_6, 1, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox_2, 2, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(scanner_form)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.dispersion_label = QtWidgets.QLabel(self.groupBox)
        self.dispersion_label.setObjectName("dispersion_label")
        self.horizontalLayout_2.addWidget(self.dispersion_label)
        self.dispersion_setpoint_combo_box = QtWidgets.QComboBox(self.groupBox)
        self.dispersion_setpoint_combo_box.setObjectName("dispersion_setpoint_combo_box")
        self.horizontalLayout_2.addWidget(self.dispersion_setpoint_combo_box)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.beta_label = QtWidgets.QLabel(self.groupBox)
        self.beta_label.setEnabled(True)
        self.beta_label.setObjectName("beta_label")
        self.horizontalLayout_10.addWidget(self.beta_label)
        self.beta_setpoint_combo_box = QtWidgets.QComboBox(self.groupBox)
        self.beta_setpoint_combo_box.setEnabled(True)
        self.beta_setpoint_combo_box.setObjectName("beta_setpoint_combo_box")
        self.horizontalLayout_10.addWidget(self.beta_setpoint_combo_box)
        self.horizontalLayout_11.addLayout(self.horizontalLayout_10)
        self.apply_optics_button = QtWidgets.QPushButton(self.groupBox)
        self.apply_optics_button.setObjectName("apply_optics_button")
        self.horizontalLayout_11.addWidget(self.apply_optics_button)
        self.gridLayout.addLayout(self.horizontalLayout_11, 0, 0, 1, 1)
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.load_quad_scan_button = QtWidgets.QPushButton(self.groupBox)
        self.load_quad_scan_button.setObjectName("load_quad_scan_button")
        self.horizontalLayout_9.addWidget(self.load_quad_scan_button)
        self.quad_scan_filename = QtWidgets.QLabel(self.groupBox)
        self.quad_scan_filename.setObjectName("quad_scan_filename")
        self.horizontalLayout_9.addWidget(self.quad_scan_filename)
        self.gridLayout.addLayout(self.horizontalLayout_9, 1, 0, 1, 1)
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.show_optics_button = QtWidgets.QPushButton(self.groupBox)
        self.show_optics_button.setObjectName("show_optics_button")
        self.horizontalLayout_13.addWidget(self.show_optics_button)
        self.cycle_quads_button = QtWidgets.QPushButton(self.groupBox)
        self.cycle_quads_button.setEnabled(True)
        self.cycle_quads_button.setObjectName("cycle_quads_button")
        self.horizontalLayout_13.addWidget(self.cycle_quads_button)
        self.open_jddd_screen_gui_button = QtWidgets.QPushButton(self.groupBox)
        self.open_jddd_screen_gui_button.setObjectName("open_jddd_screen_gui_button")
        self.horizontalLayout_13.addWidget(self.open_jddd_screen_gui_button)
        self.gridLayout.addLayout(self.horizontalLayout_13, 2, 0, 1, 1)
        self.gridLayout_4.addWidget(self.groupBox, 0, 0, 1, 1)

        self.retranslateUi(scanner_form)
        QtCore.QMetaObject.connectSlotsByName(scanner_form)

    def retranslateUi(self, scanner_form):
        _translate = QtCore.QCoreApplication.translate
        scanner_form.setWindowTitle(_translate("scanner_form", "Form"))
        self.groupBox_3.setTitle(_translate("scanner_form", "Scan Options"))
        self.pick_slice_based_on_position_checkbox.setText(_translate("scanner_form", "Pick Slice Based On Position"))
        self.do_beta_scan_checkbox.setText(_translate("scanner_form", "Do Beta Scan"))
        self.remeasure_dispersoin_checkbox.setText(_translate("scanner_form", "Remeasure Dispersion"))
        self.do_full_phase_space_checkbox.setText(_translate("scanner_form", "Full Phase Space"))
        self.dscan_voltage_label.setText(_translate("scanner_form", "Dispersion Scan TDS Voltage / MV"))
        self.tds_voltages_label.setText(_translate("scanner_form", "TDS Scan Voltages / MV"))
        self.label.setText(_translate("scanner_form", "Normalised Distance from Centre (±0.5)"))
        self.groupBox_2.setTitle(_translate("scanner_form", "Measurement Control"))
        self.background_shots_label.setText(_translate("scanner_form", "Background shots "))
        self.beam_shots_label.setText(_translate("scanner_form", "Beam Shots"))
        self.start_measurement_button.setText(_translate("scanner_form", "Start Measurement"))
        self.stop_measurement_button.setText(_translate("scanner_form", "Cancel"))
        self.preferences_button.setText(_translate("scanner_form", "Configure..."))
        self.measurement_name_label.setText(_translate("scanner_form", "Measurement Slug"))
        self.groupBox.setTitle(_translate("scanner_form", "Optics"))
        self.dispersion_label.setText(_translate("scanner_form", "<html><head/><body><p>Screen <span style=\" font-style:italic;\">D</span><span style=\" font-style:italic; vertical-align:sub;\">x </span>/ m</p></body></html>"))
        self.beta_label.setText(_translate("scanner_form", "<html><head/><body><p>Screen <span style=\" font-style:italic;\">β</span><span style=\" font-style:italic; vertical-align:sub;\">x </span>/ m</p></body></html>"))
        self.apply_optics_button.setText(_translate("scanner_form", "Apply"))
        self.load_quad_scan_button.setText(_translate("scanner_form", "Load Quad Scan..."))
        self.quad_scan_filename.setText(_translate("scanner_form", "No File"))
        self.show_optics_button.setText(_translate("scanner_form", "Optics At Screen"))
        self.cycle_quads_button.setText(_translate("scanner_form", "Cycle Quads"))
        self.open_jddd_screen_gui_button.setText(_translate("scanner_form", "Open Camera JDDD..."))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    scanner_form = QtWidgets.QWidget()
    ui = Ui_scanner_form()
    ui.setupUi(scanner_form)
    scanner_form.show()
    sys.exit(app.exec_())
