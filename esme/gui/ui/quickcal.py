# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'quickcal.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(941, 724)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout.setObjectName("gridLayout")
        self.set_phase_00 = QtWidgets.QPushButton(self.groupBox)
        self.set_phase_00.setObjectName("set_phase_00")
        self.gridLayout.addWidget(self.set_phase_00, 0, 0, 1, 1)
        self.phase_00_label = QtWidgets.QLabel(self.groupBox)
        self.phase_00_label.setText("")
        self.phase_00_label.setObjectName("phase_00_label")
        self.gridLayout.addWidget(self.phase_00_label, 0, 1, 1, 1)
        self.set_phase_01 = QtWidgets.QPushButton(self.groupBox)
        self.set_phase_01.setEnabled(False)
        self.set_phase_01.setObjectName("set_phase_01")
        self.gridLayout.addWidget(self.set_phase_01, 1, 0, 1, 1)
        self.phase_01_label = QtWidgets.QLabel(self.groupBox)
        self.phase_01_label.setText("")
        self.phase_01_label.setObjectName("phase_01_label")
        self.gridLayout.addWidget(self.phase_01_label, 1, 1, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox, 0, 0, 1, 2)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.set_phase_10 = QtWidgets.QPushButton(self.groupBox_2)
        self.set_phase_10.setEnabled(False)
        self.set_phase_10.setObjectName("set_phase_10")
        self.gridLayout_2.addWidget(self.set_phase_10, 0, 0, 1, 1)
        self.phase_10_label = QtWidgets.QLabel(self.groupBox_2)
        self.phase_10_label.setText("")
        self.phase_10_label.setObjectName("phase_10_label")
        self.gridLayout_2.addWidget(self.phase_10_label, 0, 1, 1, 1)
        self.set_phase_11 = QtWidgets.QPushButton(self.groupBox_2)
        self.set_phase_11.setEnabled(False)
        self.set_phase_11.setObjectName("set_phase_11")
        self.gridLayout_2.addWidget(self.set_phase_11, 1, 0, 1, 1)
        self.phase_11_label = QtWidgets.QLabel(self.groupBox_2)
        self.phase_11_label.setText("")
        self.phase_11_label.setObjectName("phase_11_label")
        self.gridLayout_2.addWidget(self.phase_11_label, 1, 1, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_2, 1, 0, 1, 2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.readback_phase_value = QtWidgets.QLabel(self.groupBox_3)
        self.readback_phase_value.setText("")
        self.readback_phase_value.setObjectName("readback_phase_value")
        self.gridLayout_3.addWidget(self.readback_phase_value, 2, 1, 1, 1)
        self.phase_spinner = QtWidgets.QDoubleSpinBox(self.groupBox_3)
        self.phase_spinner.setObjectName("phase_spinner")
        self.gridLayout_3.addWidget(self.phase_spinner, 1, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setObjectName("label_5")
        self.gridLayout_3.addWidget(self.label_5, 2, 0, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 1, 0, 1, 1)
        self.sub_180_deg_button = QtWidgets.QPushButton(self.groupBox_3)
        self.sub_180_deg_button.setObjectName("sub_180_deg_button")
        self.gridLayout_3.addWidget(self.sub_180_deg_button, 0, 1, 1, 1)
        self.add_180_deg_button = QtWidgets.QPushButton(self.groupBox_3)
        self.add_180_deg_button.setObjectName("add_180_deg_button")
        self.gridLayout_3.addWidget(self.add_180_deg_button, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_3, 2, 0, 1, 2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_4.setObjectName("groupBox_4")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_4)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.sbm_control = SpecialBunchMidLayerPanel(self.groupBox_4)
        self.sbm_control.setObjectName("sbm_control")
        self.gridLayout_4.addWidget(self.sbm_control, 0, 0, 1, 1)
        self.gridLayout_5.addWidget(self.groupBox_4, 3, 0, 1, 2)
        self.start_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_calib_button.setEnabled(False)
        self.start_calib_button.setObjectName("start_calib_button")
        self.gridLayout_5.addWidget(self.start_calib_button, 4, 0, 1, 1)
        self.cancel_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.cancel_calib_button.setEnabled(False)
        self.cancel_calib_button.setObjectName("cancel_calib_button")
        self.gridLayout_5.addWidget(self.cancel_calib_button, 4, 1, 1, 1)
        self.calib_plot = MPLCanvas(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.calib_plot.sizePolicy().hasHeightForWidth())
        self.calib_plot.setSizePolicy(sizePolicy)
        self.calib_plot.setObjectName("calib_plot")
        self.gridLayout_5.addWidget(self.calib_plot, 0, 2, 5, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 941, 24))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox.setTitle(_translate("MainWindow", "First Phase Range"))
        self.set_phase_00.setText(_translate("MainWindow", "Set First"))
        self.set_phase_01.setText(_translate("MainWindow", "Set Second"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Second Phase Range"))
        self.set_phase_10.setText(_translate("MainWindow", "Set First"))
        self.set_phase_11.setText(_translate("MainWindow", "Set Second"))
        self.groupBox_3.setTitle(_translate("MainWindow", "TDS Phase"))
        self.label_5.setText(_translate("MainWindow", "Readback / °"))
        self.label_7.setText(_translate("MainWindow", "Setpoint / °"))
        self.sub_180_deg_button.setText(_translate("MainWindow", "-180°"))
        self.add_180_deg_button.setText(_translate("MainWindow", "+180°"))
        self.groupBox_4.setTitle(_translate("MainWindow", "Special Bunch Midlayer"))
        self.start_calib_button.setText(_translate("MainWindow", "Start Calibration"))
        self.cancel_calib_button.setText(_translate("MainWindow", "Cancel"))


from esme.gui.widgets.mpl_widget import MPLCanvas
from esme.gui.widgets.sbunchpanel import SpecialBunchMidLayerPanel

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())