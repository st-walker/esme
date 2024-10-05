# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'slicer.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_slice_analysis_gui(object):
    def setupUi(self, slice_analysis_gui):
        slice_analysis_gui.setObjectName("slice_analysis_gui")
        slice_analysis_gui.resize(1017, 676)
        self.gridLayout = QtWidgets.QGridLayout(slice_analysis_gui)
        self.gridLayout.setObjectName("gridLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(slice_analysis_gui)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.beta_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.beta_spinbox.setEnabled(False)
        self.beta_spinbox.setDecimals(3)
        self.beta_spinbox.setSingleStep(0.1)
        self.beta_spinbox.setObjectName("beta_spinbox")
        self.gridLayout_3.addWidget(self.beta_spinbox, 9, 1, 1, 1)
        self.voltage_label = QtWidgets.QLabel(self.groupBox_2)
        self.voltage_label.setEnabled(False)
        self.voltage_label.setObjectName("voltage_label")
        self.gridLayout_3.addWidget(self.voltage_label, 8, 0, 1, 1)
        self.energy_label = QtWidgets.QLabel(self.groupBox_2)
        self.energy_label.setEnabled(False)
        self.energy_label.setObjectName("energy_label")
        self.gridLayout_3.addWidget(self.energy_label, 11, 0, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.groupBox_2)
        self.label_2.setEnabled(False)
        self.label_2.setObjectName("label_2")
        self.gridLayout_3.addWidget(self.label_2, 1, 0, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.groupBox_2)
        self.label_3.setEnabled(False)
        self.label_3.setObjectName("label_3")
        self.gridLayout_3.addWidget(self.label_3, 4, 0, 1, 1)
        self.sigma_r_label = QtWidgets.QLabel(self.groupBox_2)
        self.sigma_r_label.setEnabled(False)
        self.sigma_r_label.setObjectName("sigma_r_label")
        self.gridLayout_3.addWidget(self.sigma_r_label, 7, 0, 1, 1)
        self.screen_name_label = QtWidgets.QLabel(self.groupBox_2)
        self.screen_name_label.setText("")
        self.screen_name_label.setObjectName("screen_name_label")
        self.gridLayout_3.addWidget(self.screen_name_label, 0, 1, 1, 1)
        self.line = QtWidgets.QFrame(self.groupBox_2)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_3.addWidget(self.line, 5, 0, 1, 2)
        self.voltage_spinbox = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.voltage_spinbox.setEnabled(False)
        self.voltage_spinbox.setDecimals(3)
        self.voltage_spinbox.setSingleStep(0.1)
        self.voltage_spinbox.setObjectName("voltage_spinbox")
        self.gridLayout_3.addWidget(self.voltage_spinbox, 8, 1, 1, 1)
        self.emittance_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.emittance_spinner.setEnabled(False)
        self.emittance_spinner.setDecimals(3)
        self.emittance_spinner.setMaximum(99.0)
        self.emittance_spinner.setSingleStep(0.1)
        self.emittance_spinner.setObjectName("emittance_spinner")
        self.gridLayout_3.addWidget(self.emittance_spinner, 13, 1, 1, 1)
        self.time_cal_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.time_cal_spinner.setEnabled(False)
        self.time_cal_spinner.setDecimals(3)
        self.time_cal_spinner.setMinimum(-1000.0)
        self.time_cal_spinner.setMaximum(1000.0)
        self.time_cal_spinner.setObjectName("time_cal_spinner")
        self.gridLayout_3.addWidget(self.time_cal_spinner, 1, 1, 1, 1)
        self.energy_cal_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.energy_cal_spinner.setEnabled(False)
        self.energy_cal_spinner.setDecimals(3)
        self.energy_cal_spinner.setObjectName("energy_cal_spinner")
        self.gridLayout_3.addWidget(self.energy_cal_spinner, 4, 1, 1, 1)
        self.emittance_label = QtWidgets.QLabel(self.groupBox_2)
        self.emittance_label.setEnabled(False)
        self.emittance_label.setObjectName("emittance_label")
        self.gridLayout_3.addWidget(self.emittance_label, 13, 0, 1, 1)
        self.screen_resolution_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.screen_resolution_spinner.setEnabled(False)
        self.screen_resolution_spinner.setDecimals(3)
        self.screen_resolution_spinner.setSingleStep(0.1)
        self.screen_resolution_spinner.setProperty("value", 24.0)
        self.screen_resolution_spinner.setObjectName("screen_resolution_spinner")
        self.gridLayout_3.addWidget(self.screen_resolution_spinner, 7, 1, 1, 1)
        self.beam_energy_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.beam_energy_spinner.setEnabled(False)
        self.beam_energy_spinner.setDecimals(3)
        self.beam_energy_spinner.setMaximum(3000.0)
        self.beam_energy_spinner.setSingleStep(0.1)
        self.beam_energy_spinner.setObjectName("beam_energy_spinner")
        self.gridLayout_3.addWidget(self.beam_energy_spinner, 11, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.groupBox_2)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 0, 0, 1, 1)
        self.beta_label = QtWidgets.QLabel(self.groupBox_2)
        self.beta_label.setEnabled(False)
        self.beta_label.setObjectName("beta_label")
        self.gridLayout_3.addWidget(self.beta_label, 9, 0, 1, 1)
        self.dispersion_label = QtWidgets.QLabel(self.groupBox_2)
        self.dispersion_label.setObjectName("dispersion_label")
        self.gridLayout_3.addWidget(self.dispersion_label, 6, 0, 1, 1)
        self.dispersion_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_2)
        self.dispersion_spinner.setDecimals(3)
        self.dispersion_spinner.setMinimum(-99.0)
        self.dispersion_spinner.setSingleStep(0.1)
        self.dispersion_spinner.setObjectName("dispersion_spinner")
        self.gridLayout_3.addWidget(self.dispersion_spinner, 6, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox_2, 0, 0, 1, 1)
        self.phase_space_canvas = MPLCanvas(slice_analysis_gui)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.phase_space_canvas.sizePolicy().hasHeightForWidth()
        )
        self.phase_space_canvas.setSizePolicy(sizePolicy)
        self.phase_space_canvas.setObjectName("phase_space_canvas")
        self.gridLayout.addWidget(self.phase_space_canvas, 0, 1, 3, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(slice_analysis_gui)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_line_edit = QtWidgets.QLineEdit(self.groupBox_3)
        self.label_line_edit.setText("")
        self.label_line_edit.setObjectName("label_line_edit")
        self.gridLayout_4.addWidget(self.label_line_edit, 6, 1, 1, 1)
        self.image_number_spinner = QtWidgets.QSpinBox(self.groupBox_3)
        self.image_number_spinner.setObjectName("image_number_spinner")
        self.gridLayout_4.addWidget(self.image_number_spinner, 3, 1, 1, 1)
        self.label_4 = QtWidgets.QLabel(self.groupBox_3)
        self.label_4.setObjectName("label_4")
        self.gridLayout_4.addWidget(self.label_4, 6, 0, 1, 1)
        self.display_image_only_button = QtWidgets.QPushButton(self.groupBox_3)
        self.display_image_only_button.setObjectName("display_image_only_button")
        self.gridLayout_4.addWidget(self.display_image_only_button, 5, 2, 1, 2)
        self.update_label_button = QtWidgets.QPushButton(self.groupBox_3)
        self.update_label_button.setObjectName("update_label_button")
        self.gridLayout_4.addWidget(self.update_label_button, 6, 2, 1, 2)
        self.line_4 = QtWidgets.QFrame(self.groupBox_3)
        self.line_4.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_4.addWidget(self.line_4, 2, 0, 1, 4)
        self.plot_selected_image_button = QtWidgets.QPushButton(self.groupBox_3)
        self.plot_selected_image_button.setObjectName("plot_selected_image_button")
        self.gridLayout_4.addWidget(self.plot_selected_image_button, 3, 2, 1, 1)
        self.label = QtWidgets.QLabel(self.groupBox_3)
        self.label.setObjectName("label")
        self.gridLayout_4.addWidget(self.label, 3, 0, 1, 1)
        self.remove_selected_image_button = QtWidgets.QPushButton(self.groupBox_3)
        self.remove_selected_image_button.setObjectName("remove_selected_image_button")
        self.gridLayout_4.addWidget(self.remove_selected_image_button, 3, 3, 1, 1)
        self.plot_all_button = QtWidgets.QPushButton(self.groupBox_3)
        self.plot_all_button.setObjectName("plot_all_button")
        self.gridLayout_4.addWidget(self.plot_all_button, 1, 0, 1, 2)
        self.clear_plots_button = QtWidgets.QPushButton(self.groupBox_3)
        self.clear_plots_button.setObjectName("clear_plots_button")
        self.gridLayout_4.addWidget(self.clear_plots_button, 1, 2, 1, 2)
        self.gridLayout.addWidget(self.groupBox_3, 1, 0, 1, 1)
        self.groupBox = QtWidgets.QGroupBox(slice_analysis_gui)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.line_2 = QtWidgets.QFrame(self.groupBox)
        self.line_2.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_2.addWidget(self.line_2, 2, 0, 1, 2)
        self.load_image_from_file_button = QtWidgets.QPushButton(self.groupBox)
        self.load_image_from_file_button.setObjectName("load_image_from_file_button")
        self.gridLayout_2.addWidget(self.load_image_from_file_button, 0, 0, 1, 2)
        self.append_image_button = QtWidgets.QPushButton(self.groupBox)
        self.append_image_button.setObjectName("append_image_button")
        self.gridLayout_2.addWidget(self.append_image_button, 3, 1, 1, 1)
        self.new_image_button = QtWidgets.QPushButton(self.groupBox)
        self.new_image_button.setObjectName("new_image_button")
        self.gridLayout_2.addWidget(self.new_image_button, 3, 0, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.groupBox)
        self.line_3.setFrameShape(QtWidgets.QFrame.HLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_2.addWidget(self.line_3, 5, 0, 1, 2)
        self.send_to_logbook_button = QtWidgets.QPushButton(self.groupBox)
        self.send_to_logbook_button.setObjectName("send_to_logbook_button")
        self.gridLayout_2.addWidget(self.send_to_logbook_button, 6, 0, 1, 1)
        self.cancel_button = QtWidgets.QPushButton(self.groupBox)
        self.cancel_button.setObjectName("cancel_button")
        self.gridLayout_2.addWidget(self.cancel_button, 6, 1, 1, 1)
        self.images_stored_label = QtWidgets.QLabel(self.groupBox)
        self.images_stored_label.setObjectName("images_stored_label")
        self.gridLayout_2.addWidget(self.images_stored_label, 4, 0, 1, 1)
        self.clear_cache_button = QtWidgets.QPushButton(self.groupBox)
        self.clear_cache_button.setObjectName("clear_cache_button")
        self.gridLayout_2.addWidget(self.clear_cache_button, 4, 1, 1, 1)
        self.gridLayout.addWidget(self.groupBox, 2, 0, 2, 1)

        self.retranslateUi(slice_analysis_gui)
        QtCore.QMetaObject.connectSlotsByName(slice_analysis_gui)

    def retranslateUi(self, slice_analysis_gui):
        _translate = QtCore.QCoreApplication.translate
        slice_analysis_gui.setWindowTitle(_translate("slice_analysis_gui", "Form"))
        self.groupBox_2.setTitle(
            _translate("slice_analysis_gui", "Measurement Parameters")
        )
        self.voltage_label.setText(
            _translate("slice_analysis_gui", "<i>V</i><sub>TDS</sub> / MV")
        )
        self.energy_label.setText(_translate("slice_analysis_gui", "<i>E</i> / MeV"))
        self.label_2.setText(
            _translate("slice_analysis_gui", "Time Calibration / µmps<sup>-1</sup>")
        )
        self.label_3.setText(
            _translate("slice_analysis_gui", "Energy Calibration / MeVm⁻¹")
        )
        self.sigma_r_label.setToolTip(
            _translate("slice_analysis_gui", "Screen Resolution")
        )
        self.sigma_r_label.setText(
            _translate("slice_analysis_gui", "<i>σ<sub>R</sub></i> / µm")
        )
        self.emittance_label.setText(
            _translate("slice_analysis_gui", "<i>ε</i><sub>n</sub> / mm·mrad")
        )
        self.screen_resolution_spinner.setToolTip(
            _translate("slice_analysis_gui", "Screen Resolution")
        )
        self.label_8.setText(_translate("slice_analysis_gui", "Screen Name"))
        self.beta_label.setText(
            _translate("slice_analysis_gui", "𝛽<sub><i>x</i></sub> / m")
        )
        self.dispersion_label.setText(
            _translate("slice_analysis_gui", "<i>D</i><sub><i>x</i></sub> / m")
        )
        self.groupBox_3.setTitle(_translate("slice_analysis_gui", "Plotting"))
        self.label_4.setText(_translate("slice_analysis_gui", "Label"))
        self.display_image_only_button.setText(
            _translate("slice_analysis_gui", "Show Image without Analysis")
        )
        self.update_label_button.setText(
            _translate("slice_analysis_gui", "Update Label")
        )
        self.plot_selected_image_button.setText(
            _translate("slice_analysis_gui", "Analyse Image")
        )
        self.label.setText(_translate("slice_analysis_gui", "Select Image"))
        self.remove_selected_image_button.setText(
            _translate("slice_analysis_gui", "Remove Analysis")
        )
        self.plot_all_button.setText(
            _translate("slice_analysis_gui", "Analyse All Images")
        )
        self.clear_plots_button.setText(
            _translate("slice_analysis_gui", "Clear All Analyses")
        )
        self.groupBox.setTitle(_translate("slice_analysis_gui", "I/O"))
        self.load_image_from_file_button.setText(
            _translate("slice_analysis_gui", "Load Image Collection From File...")
        )
        self.append_image_button.setText(
            _translate("slice_analysis_gui", "Append Image From Screen")
        )
        self.new_image_button.setText(
            _translate("slice_analysis_gui", "New Image From Screen")
        )
        self.send_to_logbook_button.setText(
            _translate("slice_analysis_gui", "Send to XFEL e-Logbook...")
        )
        self.cancel_button.setText(_translate("slice_analysis_gui", "Cancel"))
        self.images_stored_label.setText(
            _translate("slice_analysis_gui", "Images Stored:")
        )
        self.clear_cache_button.setText(_translate("slice_analysis_gui", "Clear"))


from esme.gui.widgets.mpl_widget import MPLCanvas

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    slice_analysis_gui = QtWidgets.QWidget()
    ui = Ui_slice_analysis_gui()
    ui.setupUi(slice_analysis_gui)
    slice_analysis_gui.show()
    sys.exit(app.exec_())
