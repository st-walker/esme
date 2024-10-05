# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'optics.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(312, 205)
        self.verticalLayout = QtWidgets.QVBoxLayout(Form)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_7 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_7.setObjectName("horizontalLayout_7")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.background_shots_label = QtWidgets.QLabel(Form)
        self.background_shots_label.setObjectName("background_shots_label")
        self.horizontalLayout_4.addWidget(self.background_shots_label)
        self.bg_shots_spinner = QtWidgets.QSpinBox(Form)
        self.bg_shots_spinner.setProperty("value", 5)
        self.bg_shots_spinner.setObjectName("bg_shots_spinner")
        self.horizontalLayout_4.addWidget(self.bg_shots_spinner)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_4)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.beam_shots_label = QtWidgets.QLabel(Form)
        self.beam_shots_label.setObjectName("beam_shots_label")
        self.horizontalLayout_5.addWidget(self.beam_shots_label)
        self.beam_shots_spinner = QtWidgets.QSpinBox(Form)
        self.beam_shots_spinner.setProperty("value", 30)
        self.beam_shots_spinner.setObjectName("beam_shots_spinner")
        self.horizontalLayout_5.addWidget(self.beam_shots_spinner)
        self.horizontalLayout_7.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addLayout(self.horizontalLayout_7)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.dispersion_setpoint_combo_box = QtWidgets.QComboBox(Form)
        self.dispersion_setpoint_combo_box.setObjectName(
            "dispersion_setpoint_combo_box"
        )
        self.horizontalLayout_2.addWidget(self.dispersion_setpoint_combo_box)
        self.pushButton = QtWidgets.QPushButton(Form)
        self.pushButton.setObjectName("pushButton")
        self.horizontalLayout_2.addWidget(self.pushButton)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout.addWidget(self.label_2)
        self.tds_amplitudes = QtWidgets.QLineEdit(Form)
        self.tds_amplitudes.setObjectName("tds_amplitudes")
        self.horizontalLayout.addWidget(self.tds_amplitudes)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.measurement_name_label = QtWidgets.QLabel(Form)
        self.measurement_name_label.setObjectName("measurement_name_label")
        self.horizontalLayout_6.addWidget(self.measurement_name_label)
        self.lineEdit = QtWidgets.QLineEdit(Form)
        self.lineEdit.setText("")
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_6.addWidget(self.lineEdit)
        self.verticalLayout.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.start_measurement_button = QtWidgets.QPushButton(Form)
        self.start_measurement_button.setCheckable(False)
        self.start_measurement_button.setAutoDefault(False)
        self.start_measurement_button.setDefault(False)
        self.start_measurement_button.setFlat(False)
        self.start_measurement_button.setObjectName("start_measurement_button")
        self.horizontalLayout_3.addWidget(self.start_measurement_button)
        self.measurement_progress_bar = QtWidgets.QProgressBar(Form)
        self.measurement_progress_bar.setProperty("value", 0)
        self.measurement_progress_bar.setInvertedAppearance(False)
        self.measurement_progress_bar.setObjectName("measurement_progress_bar")
        self.horizontalLayout_3.addWidget(self.measurement_progress_bar)
        self.verticalLayout.addLayout(self.horizontalLayout_3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.background_shots_label.setText(_translate("Form", "Background shots "))
        self.beam_shots_label.setText(_translate("Form", "Beam Shots"))
        self.pushButton.setText(_translate("Form", "Apply Optics"))
        self.label_2.setText(_translate("Form", "TDS Scan Amplitudes"))
        self.tds_amplitudes.setText(_translate("Form", "8, 10, 12, 14, 16"))
        self.measurement_name_label.setText(_translate("Form", "Measurement Slug"))
        self.start_measurement_button.setText(_translate("Form", "Start Measurement"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())
