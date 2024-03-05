# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scanner_config.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(587, 640)
        self.gridLayout = QtWidgets.QGridLayout(Dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.gridLayout_2.addWidget(self.label_2, 1, 0, 1, 1)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 0, 0, 1, 1)
        self.output_directory_lineedit = QtWidgets.QLineEdit(Dialog)
        self.output_directory_lineedit.setObjectName("output_directory_lineedit")
        self.gridLayout_2.addWidget(self.output_directory_lineedit, 3, 1, 1, 1)
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setObjectName("label_3")
        self.gridLayout_2.addWidget(self.label_3, 2, 0, 1, 1)
        self.label_5 = QtWidgets.QLabel(Dialog)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 3, 0, 1, 1)
        self.quad_sleep_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.quad_sleep_spinbox.setObjectName("quad_sleep_spinbox")
        self.gridLayout_2.addWidget(self.quad_sleep_spinbox, 1, 1, 1, 1)
        self.beam_on_wait_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.beam_on_wait_spinbox.setObjectName("beam_on_wait_spinbox")
        self.gridLayout_2.addWidget(self.beam_on_wait_spinbox, 2, 1, 1, 1)
        self.tds_amplitude_wait_spinbox = QtWidgets.QDoubleSpinBox(Dialog)
        self.tds_amplitude_wait_spinbox.setObjectName("tds_amplitude_wait_spinbox")
        self.gridLayout_2.addWidget(self.tds_amplitude_wait_spinbox, 0, 1, 1, 1)
        self.gridLayout.addLayout(self.gridLayout_2, 0, 0, 1, 1)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setCenterButtons(False)
        self.buttonBox.setObjectName("buttonBox")
        self.gridLayout.addWidget(self.buttonBox, 1, 0, 1, 1)

        self.retranslateUi(Dialog)
        self.buttonBox.rejected.connect(Dialog.reject) # type: ignore
        self.buttonBox.accepted.connect(Dialog.accept) # type: ignore
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label_2.setText(_translate("Dialog", "Sleep after Quad Set"))
        self.label.setText(_translate("Dialog", "Sleep after TDS Amplitude Set"))
        self.label_3.setText(_translate("Dialog", "Beam On Wait"))
        self.label_5.setText(_translate("Dialog", "Data Directory"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())