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
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.amp_voltage_plot_widget = MatplotlibCanvas(self.centralwidget)
        self.amp_voltage_plot_widget.setObjectName("amp_voltage_plot_widget")
        self.horizontalLayout_2.addWidget(self.amp_voltage_plot_widget)
        self.phase_com_plot_widget = MatplotlibCanvas(self.centralwidget)
        self.phase_com_plot_widget.setObjectName("phase_com_plot_widget")
        self.horizontalLayout_2.addWidget(self.phase_com_plot_widget)
        self.gridLayout.addLayout(self.horizontalLayout_2, 0, 0, 2, 2)
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
        self.gridLayout.addLayout(self.verticalLayout, 1, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.amplitudes_line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.amplitudes_line_edit.setObjectName("amplitudes_line_edit")
        self.horizontalLayout.addWidget(self.amplitudes_line_edit)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 2, 1, 1)
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
        self.label.setText(_translate("MainWindow", "Amplitudes / %"))
        self.amplitudes_line_edit.setText(_translate("MainWindow", "5, 10, 15, 20"))
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
