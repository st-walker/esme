# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibration.ui'
#
# Created by: PyQt5 UI code generator 5.15.9
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.amp_voltage_plot_widget = MatplotlibCanvas(self.centralwidget)
        self.amp_voltage_plot_widget.setObjectName("amp_voltage_plot_widget")
        self.horizontalLayout_2.addWidget(self.amp_voltage_plot_widget)
        self.phase_com_plot_widget = MatplotlibCanvas(self.centralwidget)
        self.phase_com_plot_widget.setObjectName("phase_com_plot_widget")
        self.horizontalLayout_2.addWidget(self.phase_com_plot_widget)
        self.verticalLayout_2.addLayout(self.horizontalLayout_2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.start_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_calib_button.setObjectName("start_calib_button")
        self.verticalLayout.addWidget(self.start_calib_button)
        self.load_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_calib_button.setObjectName("load_calib_button")
        self.verticalLayout.addWidget(self.load_calib_button)
        self.apply_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.apply_calib_button.setObjectName("apply_calib_button")
        self.verticalLayout.addWidget(self.apply_calib_button)
        self.screen_selection_box = QtWidgets.QComboBox(self.centralwidget)
        self.screen_selection_box.setObjectName("screen_selection_box")
        self.verticalLayout.addWidget(self.screen_selection_box)
        self.horizontalLayout.addLayout(self.verticalLayout)
        self.calib_textedit = QtWidgets.QTextEdit(self.centralwidget)
        self.calib_textedit.setMaximumSize(QtCore.QSize(246, 69))
        self.calib_textedit.setObjectName("calib_textedit")
        self.horizontalLayout.addWidget(self.calib_textedit)
        self.verticalLayout_2.addLayout(self.horizontalLayout)
        self.gridLayout.addLayout(self.verticalLayout_2, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setDefaultUp(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setTearOffEnabled(False)
        self.menuMenu.setObjectName("menuMenu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_Print_to_Logbook = QtWidgets.QAction(MainWindow)
        self.action_Print_to_Logbook.setObjectName("action_Print_to_Logbook")
        self.action_Quit = QtWidgets.QAction(MainWindow)
        self.action_Quit.setObjectName("action_Quit")
        self.menuMenu.addAction(self.action_Print_to_Logbook)
        self.menuMenu.addAction(self.action_Quit)
        self.menubar.addAction(self.menuMenu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.start_calib_button.setText(_translate("MainWindow", "Start"))
        self.load_calib_button.setText(_translate("MainWindow", "Load..."))
        self.apply_calib_button.setText(_translate("MainWindow", "Apply"))
        self.calib_textedit.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><meta charset=\"utf-8\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"hr { height: 1px; border-width: 0; }\n"
"li.unchecked::marker { content: \"\\2610\"; }\n"
"li.checked::marker { content: \"\\2612\"; }\n"
"</style></head><body style=\" font-family:\'.AppleSystemUIFont\'; font-size:13pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">phi0 = [-30, -20, 10, 5]</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">phi1 = [10, 20, 30, 50]</p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">amps = [5, 10, 15, 20]</p></body></html>"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.action_Print_to_Logbook.setText(_translate("MainWindow", "&Print to Logbook"))
        self.action_Quit.setText(_translate("MainWindow", "&Quit"))
from esme.gui.mpl_widget import MatplotlibCanvas


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
