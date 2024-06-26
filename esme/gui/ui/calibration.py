# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'calibration.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 627)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.centre_of_mass_with_phase_plot = GraphicsLayoutWidget(self.centralwidget)
        self.centre_of_mass_with_phase_plot.setMinimumSize(QtCore.QSize(289, 169))
        self.centre_of_mass_with_phase_plot.setObjectName("centre_of_mass_with_phase_plot")
        self.verticalLayout_3.addWidget(self.centre_of_mass_with_phase_plot)
        self.processed_image_plot = GraphicsLayoutWidget(self.centralwidget)
        self.processed_image_plot.setMinimumSize(QtCore.QSize(289, 170))
        self.processed_image_plot.setObjectName("processed_image_plot")
        self.verticalLayout_3.addWidget(self.processed_image_plot)
        self.gridLayout.addLayout(self.verticalLayout_3, 0, 0, 1, 1)
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.final_calibration_plot = MatplotlibCanvas(self.centralwidget)
        self.final_calibration_plot.setMinimumSize(QtCore.QSize(289, 170))
        self.final_calibration_plot.setObjectName("final_calibration_plot")
        self.verticalLayout_4.addWidget(self.final_calibration_plot)
        self.zero_crossing_extraction_plot = MatplotlibCanvas(self.centralwidget)
        self.zero_crossing_extraction_plot.setMinimumSize(QtCore.QSize(289, 170))
        self.zero_crossing_extraction_plot.setObjectName("zero_crossing_extraction_plot")
        self.verticalLayout_4.addWidget(self.zero_crossing_extraction_plot)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.amplitudes_line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.amplitudes_line_edit.setObjectName("amplitudes_line_edit")
        self.horizontalLayout.addWidget(self.amplitudes_line_edit)
        self.verticalLayout_4.addLayout(self.horizontalLayout)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.screen_name_line_edit = QtWidgets.QLineEdit(self.centralwidget)
        self.screen_name_line_edit.setObjectName("screen_name_line_edit")
        self.verticalLayout.addWidget(self.screen_name_line_edit)
        self.start_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_calib_button.setMaximumSize(QtCore.QSize(220, 16777215))
        self.start_calib_button.setObjectName("start_calib_button")
        self.verticalLayout.addWidget(self.start_calib_button)
        self.load_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.load_calib_button.setObjectName("load_calib_button")
        self.verticalLayout.addWidget(self.load_calib_button)
        self.apply_calib_button = QtWidgets.QPushButton(self.centralwidget)
        self.apply_calib_button.setObjectName("apply_calib_button")
        self.verticalLayout.addWidget(self.apply_calib_button)
        self.verticalLayout_4.addLayout(self.verticalLayout)
        self.gridLayout.addLayout(self.verticalLayout_4, 0, 1, 1, 1)
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
        self.label.setText(_translate("MainWindow", "Amplitudes / %"))
        self.amplitudes_line_edit.setText(_translate("MainWindow", "5, 10, 15, 20"))
        self.start_calib_button.setText(_translate("MainWindow", "Start"))
        self.load_calib_button.setText(_translate("MainWindow", "Load..."))
        self.apply_calib_button.setText(_translate("MainWindow", "Apply"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.action_Print_to_Logbook.setText(_translate("MainWindow", "&Print to Logbook"))
        self.action_Quit.setText(_translate("MainWindow", "&Quit"))
from esme.gui.widgets.mpl_widget import MatplotlibCanvas
from pyqtgraph import GraphicsLayoutWidget


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
