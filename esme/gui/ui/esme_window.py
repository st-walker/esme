# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'esme_window.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(802, 621)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout.addWidget(self.line, 0, 1, 1, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.measurement_log_browser = QtWidgets.QTextBrowser(self.centralwidget)
        self.measurement_log_browser.setObjectName("measurement_log_browser")
        self.verticalLayout_3.addWidget(self.measurement_log_browser)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.measurement_name_label = QtWidgets.QLabel(self.groupBox)
        self.measurement_name_label.setObjectName("measurement_name_label")
        self.horizontalLayout_6.addWidget(self.measurement_name_label)
        self.lineEdit = QtWidgets.QLineEdit(self.groupBox)
        self.lineEdit.setText("")
        self.lineEdit.setClearButtonEnabled(False)
        self.lineEdit.setObjectName("lineEdit")
        self.horizontalLayout_6.addWidget(self.lineEdit)
        self.save_data_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.save_data_checkbox.setChecked(False)
        self.save_data_checkbox.setTristate(False)
        self.save_data_checkbox.setObjectName("save_data_checkbox")
        self.horizontalLayout_6.addWidget(self.save_data_checkbox)
        self.gridLayout_6.addLayout(self.horizontalLayout_6, 1, 0, 1, 2)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.location_label = QtWidgets.QLabel(self.groupBox)
        self.location_label.setObjectName("location_label")
        self.horizontalLayout_5.addWidget(self.location_label)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.i1_radio_button = QtWidgets.QRadioButton(self.groupBox)
        self.i1_radio_button.setChecked(True)
        self.i1_radio_button.setObjectName("i1_radio_button")
        self.horizontalLayout_4.addWidget(self.i1_radio_button)
        self.b2_radio_buton = QtWidgets.QRadioButton(self.groupBox)
        self.b2_radio_buton.setCheckable(False)
        self.b2_radio_buton.setObjectName("b2_radio_buton")
        self.horizontalLayout_4.addWidget(self.b2_radio_buton)
        self.horizontalLayout_5.addLayout(self.horizontalLayout_4)
        self.verticalLayout.addLayout(self.horizontalLayout_5)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.beam_shots_spinner = QtWidgets.QSpinBox(self.groupBox)
        self.beam_shots_spinner.setProperty("value", 30)
        self.beam_shots_spinner.setObjectName("beam_shots_spinner")
        self.gridLayout_3.addWidget(self.beam_shots_spinner, 0, 1, 1, 1)
        self.background_shots_label = QtWidgets.QLabel(self.groupBox)
        self.background_shots_label.setObjectName("background_shots_label")
        self.gridLayout_3.addWidget(self.background_shots_label, 1, 0, 1, 1)
        self.beam_shots_label = QtWidgets.QLabel(self.groupBox)
        self.beam_shots_label.setObjectName("beam_shots_label")
        self.gridLayout_3.addWidget(self.beam_shots_label, 0, 0, 1, 1)
        self.bg_shots_spinner = QtWidgets.QSpinBox(self.groupBox)
        self.bg_shots_spinner.setProperty("value", 5)
        self.bg_shots_spinner.setObjectName("bg_shots_spinner")
        self.gridLayout_3.addWidget(self.bg_shots_spinner, 1, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout_3)
        self.gridLayout_6.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.start_measurement_button = QtWidgets.QPushButton(self.groupBox)
        self.start_measurement_button.setCheckable(False)
        self.start_measurement_button.setAutoDefault(False)
        self.start_measurement_button.setDefault(False)
        self.start_measurement_button.setFlat(False)
        self.start_measurement_button.setObjectName("start_measurement_button")
        self.horizontalLayout_3.addWidget(self.start_measurement_button)
        self.measurement_progress_bar = QtWidgets.QProgressBar(self.groupBox)
        self.measurement_progress_bar.setProperty("value", 0)
        self.measurement_progress_bar.setObjectName("measurement_progress_bar")
        self.horizontalLayout_3.addWidget(self.measurement_progress_bar)
        self.gridLayout_6.addLayout(self.horizontalLayout_3, 2, 0, 1, 2)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.tds_calibration_button = QtWidgets.QPushButton(self.groupBox)
        self.tds_calibration_button.setFlat(False)
        self.tds_calibration_button.setObjectName("tds_calibration_button")
        self.gridLayout_5.addWidget(self.tds_calibration_button, 1, 0, 1, 1)
        self.dispersion_setpoint_combo_box = QtWidgets.QComboBox(self.groupBox)
        self.dispersion_setpoint_combo_box.setObjectName(
            "dispersion_setpoint_combo_box"
        )
        self.gridLayout_5.addWidget(self.dispersion_setpoint_combo_box, 0, 0, 1, 1)
        self.gridLayout_6.addLayout(self.gridLayout_5, 0, 1, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 2, 1, 1)
        self.screen_widget = ImageView(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.screen_widget.sizePolicy().hasHeightForWidth()
        )
        self.screen_widget.setSizePolicy(sizePolicy)
        self.screen_widget.setObjectName("screen_widget")
        self.label = QtWidgets.QLabel(self.screen_widget)
        self.label.setGeometry(QtCore.QRect(-10, 430, 281, 51))
        self.label.setObjectName("label")
        self.textEdit = QtWidgets.QTextEdit(self.screen_widget)
        self.textEdit.setGeometry(QtCore.QRect(80, 300, 104, 74))
        self.textEdit.setObjectName("textEdit")
        self.label_4 = QtWidgets.QLabel(self.screen_widget)
        self.label_4.setGeometry(QtCore.QRect(30, 80, 281, 51))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.screen_widget)
        self.label_5.setGeometry(QtCore.QRect(30, 150, 281, 101))
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.screen_widget, 0, 0, 1, 1)
        self.gridLayout_2.addLayout(self.gridLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 802, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionClose = QtWidgets.QAction(MainWindow)
        self.actionClose.setObjectName("actionClose")
        self.actionPrint_to_Logbook = QtWidgets.QAction(MainWindow)
        self.actionPrint_to_Logbook.setObjectName("actionPrint_to_Logbook")
        self.menuMenu.addAction(self.actionPrint_to_Logbook)
        self.menuMenu.addAction(self.actionClose)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Energy Spread Measurement"))
        self.groupBox.setTitle(_translate("MainWindow", "Measurement"))
        self.measurement_name_label.setText(
            _translate("MainWindow", "Output Directory")
        )
        self.save_data_checkbox.setText(_translate("MainWindow", "Online Analysis"))
        self.location_label.setText(_translate("MainWindow", "Location"))
        self.i1_radio_button.setText(_translate("MainWindow", "I1"))
        self.b2_radio_buton.setText(_translate("MainWindow", "B2"))
        self.background_shots_label.setText(
            _translate("MainWindow", "Background shots ")
        )
        self.beam_shots_label.setText(_translate("MainWindow", "Beam Shots"))
        self.start_measurement_button.setText(
            _translate("MainWindow", "Start Measurement")
        )
        self.tds_calibration_button.setText(
            _translate("MainWindow", "TDS Calibration...")
        )
        self.label.setText(
            _translate(
                "MainWindow",
                "<html><head/><body><p>Only D<span style=\" vertical-align:sub;\">x/y</span> scan e.g. provide coefficient of TDS manually.</p></body></html>",
            )
        )
        self.textEdit.setHtml(
            _translate(
                "MainWindow",
                "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                "<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
                "p, li { white-space: pre-wrap; }\n"
                "hr { height: 1px; border-width: 0; }\n"
                "li.unchecked::marker { content: \"\\2610\"; }\n"
                "li.checked::marker { content: \"\\2612\"; }\n"
                "</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
                "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">laser heater button please...</p></body></html>",
            )
        )
        self.label_4.setText(
            _translate(
                "MainWindow",
                "<html><head/><body><p>Show pixel widths automatically please...</p></body></html>",
            )
        )
        self.label_5.setText(
            _translate(
                "MainWindow",
                "<html><head/><body><p>should be able 2 take a single datapoint.</p><p>should be able to control tDS from here too actually I think after all..</p><p>should be able to do beta scan too.</p></body></html>",
            )
        )
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.actionClose.setText(_translate("MainWindow", "Close"))
        self.actionClose.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.actionPrint_to_Logbook.setText(
            _translate("MainWindow", "Print to Logbook")
        )
        self.actionPrint_to_Logbook.setShortcut(_translate("MainWindow", "Ctrl+P"))


from pyqtgraph import ImageView

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
