# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'scan_results.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_results_box_dialog(object):
    def setupUi(self, results_box_dialog):
        results_box_dialog.setObjectName("results_box_dialog")
        results_box_dialog.resize(480, 457)
        self.gridLayout = QtWidgets.QGridLayout(results_box_dialog)
        self.gridLayout.setObjectName("gridLayout")
        self.result_text_browser = QtWidgets.QTextBrowser(results_box_dialog)
        self.result_text_browser.setReadOnly(False)
        self.result_text_browser.setAcceptRichText(False)
        self.result_text_browser.setObjectName("result_text_browser")
        self.gridLayout.addWidget(self.result_text_browser, 0, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.send_to_logbook_button = QtWidgets.QPushButton(results_box_dialog)
        self.send_to_logbook_button.setObjectName("send_to_logbook_button")
        self.horizontalLayout.addWidget(self.send_to_logbook_button)
        self.pushButton_2 = QtWidgets.QPushButton(results_box_dialog)
        self.pushButton_2.setObjectName("pushButton_2")
        self.horizontalLayout.addWidget(self.pushButton_2)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)

        self.retranslateUi(results_box_dialog)
        QtCore.QMetaObject.connectSlotsByName(results_box_dialog)

    def retranslateUi(self, results_box_dialog):
        _translate = QtCore.QCoreApplication.translate
        results_box_dialog.setWindowTitle(_translate("results_box_dialog", "Dialog"))
        self.send_to_logbook_button.setText(_translate("results_box_dialog", "Send To Logbook"))
        self.pushButton_2.setText(_translate("results_box_dialog", "Close"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    results_box_dialog = QtWidgets.QDialog()
    ui = Ui_results_box_dialog()
    ui.setupUi(results_box_dialog)
    results_box_dialog.show()
    sys.exit(app.exec_())
