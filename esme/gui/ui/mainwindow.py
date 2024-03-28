# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1400, 785)
        MainWindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setObjectName("tabWidget")
        self.maintab = QtWidgets.QWidget()
        self.maintab.setObjectName("maintab")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.maintab)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.groupBox = QtWidgets.QGroupBox(self.maintab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.nbg_spinbox = QtWidgets.QSpinBox(self.groupBox)
        self.nbg_spinbox.setMaximum(10)
        self.nbg_spinbox.setObjectName("nbg_spinbox")
        self.horizontalLayout.addWidget(self.nbg_spinbox)
        self.gridLayout_6.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.nbeam_spinbox = QtWidgets.QSpinBox(self.groupBox)
        self.nbeam_spinbox.setButtonSymbols(QtWidgets.QAbstractSpinBox.UpDownArrows)
        self.nbeam_spinbox.setMaximum(99)
        self.nbeam_spinbox.setObjectName("nbeam_spinbox")
        self.horizontalLayout_4.addWidget(self.nbeam_spinbox)
        self.gridLayout_6.addLayout(self.horizontalLayout_4, 0, 1, 1, 2)
        self.subtract_bg_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.subtract_bg_checkbox.setObjectName("subtract_bg_checkbox")
        self.gridLayout_6.addWidget(self.subtract_bg_checkbox, 1, 0, 1, 1)
        self.isolate_beam_image_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.isolate_beam_image_checkbox.setObjectName("isolate_beam_image_checkbox")
        self.gridLayout_6.addWidget(self.isolate_beam_image_checkbox, 1, 1, 1, 2)
        self.take_background_button = QtWidgets.QPushButton(self.groupBox)
        self.take_background_button.setObjectName("take_background_button")
        self.gridLayout_6.addWidget(self.take_background_button, 2, 0, 1, 2)
        self.take_data_button = QtWidgets.QPushButton(self.groupBox)
        self.take_data_button.setObjectName("take_data_button")
        self.gridLayout_6.addWidget(self.take_data_button, 2, 2, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox, 3, 1, 1, 1)
        self.groupBox_4 = QtWidgets.QGroupBox(self.maintab)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.machine_state_widget = LPSStateWatcher(self.groupBox_4)
        self.machine_state_widget.setObjectName("machine_state_widget")
        self.verticalLayout_2.addWidget(self.machine_state_widget)
        self.gridLayout_3.addWidget(self.groupBox_4, 4, 1, 1, 1)
        self.groupBox_2 = QtWidgets.QGroupBox(self.maintab)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tds_panel = TDSControl(self.groupBox_2)
        self.tds_panel.setObjectName("tds_panel")
        self.gridLayout_4.addWidget(self.tds_panel, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_2, 2, 1, 1, 1)
        self.screen_display_widget = ScreenDisplayWidget(self.maintab)
        self.screen_display_widget.setObjectName("screen_display_widget")
        self.gridLayout_3.addWidget(self.screen_display_widget, 0, 0, 5, 1)
        self.groupBox_3 = QtWidgets.QGroupBox(self.maintab)
        self.groupBox_3.setObjectName("groupBox_3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.groupBox_3)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.special_bunch_panel = SpecialBunchMidLayerPanel(self.groupBox_3)
        self.special_bunch_panel.setObjectName("special_bunch_panel")
        self.gridLayout_5.addWidget(self.special_bunch_panel, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.groupBox_3, 1, 1, 1, 1)
        self.controls_group_box = QtWidgets.QGroupBox(self.maintab)
        self.controls_group_box.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.controls_group_box.setObjectName("controls_group_box")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.controls_group_box)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.screen_label = QtWidgets.QLabel(self.controls_group_box)
        self.screen_label.setObjectName("screen_label")
        self.horizontalLayout_3.addWidget(self.screen_label)
        self.select_screen_combobox = QtWidgets.QComboBox(self.controls_group_box)
        self.select_screen_combobox.setPlaceholderText("")
        self.select_screen_combobox.setObjectName("select_screen_combobox")
        self.horizontalLayout_3.addWidget(self.select_screen_combobox)
        self.jddd_screen_gui_button = QtWidgets.QPushButton(self.controls_group_box)
        self.jddd_screen_gui_button.setObjectName("jddd_screen_gui_button")
        self.horizontalLayout_3.addWidget(self.jddd_screen_gui_button)
        self.gridLayout_2.addLayout(self.horizontalLayout_3, 1, 0, 1, 1)
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.section_label = QtWidgets.QLabel(self.controls_group_box)
        self.section_label.setObjectName("section_label")
        self.horizontalLayout_2.addWidget(self.section_label)
        self.i1_radio_button = QtWidgets.QRadioButton(self.controls_group_box)
        self.i1_radio_button.setChecked(True)
        self.i1_radio_button.setObjectName("i1_radio_button")
        self.horizontalLayout_2.addWidget(self.i1_radio_button)
        self.b2_radio_button = QtWidgets.QRadioButton(self.controls_group_box)
        self.b2_radio_button.setObjectName("b2_radio_button")
        self.horizontalLayout_2.addWidget(self.b2_radio_button)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.gridLayout_2.addLayout(self.verticalLayout, 0, 0, 1, 1)
        self.gridLayout_3.addWidget(self.controls_group_box, 0, 1, 1, 1)
        self.tabWidget.addTab(self.maintab, "")
        self.logtab = QtWidgets.QWidget()
        self.logtab.setObjectName("logtab")
        self.gridLayout = QtWidgets.QGridLayout(self.logtab)
        self.gridLayout.setObjectName("gridLayout")
        self.measurement_log_browser = QtWidgets.QTextBrowser(self.logtab)
        self.measurement_log_browser.setObjectName("measurement_log_browser")
        self.gridLayout.addWidget(self.measurement_log_browser, 0, 0, 1, 1)
        self.tabWidget.addTab(self.logtab, "")
        self.gridLayout_9.addWidget(self.tabWidget, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1400, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuOptics = QtWidgets.QMenu(self.menubar)
        self.menuOptics.setObjectName("menuOptics")
        MainWindow.setMenuBar(self.menubar)
        self.action_close = QtWidgets.QAction(MainWindow)
        self.action_close.setObjectName("action_close")
        self.action_print_to_logbook = QtWidgets.QAction(MainWindow)
        self.action_print_to_logbook.setObjectName("action_print_to_logbook")
        self.actionBunch_Length = QtWidgets.QAction(MainWindow)
        self.actionBunch_Length.setObjectName("actionBunch_Length")
        self.actionLongitudinal_Phase_Space = QtWidgets.QAction(MainWindow)
        self.actionLongitudinal_Phase_Space.setObjectName("actionLongitudinal_Phase_Space")
        self.actionEmittance = QtWidgets.QAction(MainWindow)
        self.actionEmittance.setObjectName("actionEmittance")
        self.actionEmail_Maintainer = QtWidgets.QAction(MainWindow)
        self.actionEmail_Maintainer.setObjectName("actionEmail_Maintainer")
        self.menuMenu.addAction(self.action_print_to_logbook)
        self.menuMenu.addAction(self.action_close)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuOptics.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Diagnostic Sections Utility"))
        self.groupBox.setTitle(_translate("MainWindow", "Image Acquisition"))
        self.label.setText(_translate("MainWindow", "Background Shots"))
        self.label_2.setText(_translate("MainWindow", "Beam Shots"))
        self.subtract_bg_checkbox.setText(_translate("MainWindow", "Subtract background"))
        self.isolate_beam_image_checkbox.setText(_translate("MainWindow", "Isolate beam image"))
        self.take_background_button.setText(_translate("MainWindow", "Accumulate Background"))
        self.take_data_button.setText(_translate("MainWindow", "Take Data..."))
        self.groupBox_4.setTitle(_translate("MainWindow", "Machine State"))
        self.groupBox_2.setTitle(_translate("MainWindow", "Transverse Deflecting Structure"))
        self.groupBox_3.setTitle(_translate("MainWindow", "Special Bunch Midlayer"))
        self.controls_group_box.setTitle(_translate("MainWindow", "Diagnostic Area"))
        self.screen_label.setText(_translate("MainWindow", "Screen"))
        self.select_screen_combobox.setToolTip(_translate("MainWindow", "The name of the screen to use"))
        self.jddd_screen_gui_button.setToolTip(_translate("MainWindow", "Open the corresponding JDDD camera control panel for the selected screen"))
        self.jddd_screen_gui_button.setText(_translate("MainWindow", "JDDD..."))
        self.section_label.setText(_translate("MainWindow", "Section"))
        self.i1_radio_button.setText(_translate("MainWindow", "I1"))
        self.b2_radio_button.setText(_translate("MainWindow", "B2"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.maintab), _translate("MainWindow", "Main"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.logtab), _translate("MainWindow", "Log"))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu"))
        self.menuOptics.setTitle(_translate("MainWindow", "Optics"))
        self.action_close.setText(_translate("MainWindow", "Close"))
        self.action_close.setShortcut(_translate("MainWindow", "Ctrl+Q"))
        self.action_print_to_logbook.setText(_translate("MainWindow", "Print to Logbook"))
        self.action_print_to_logbook.setShortcut(_translate("MainWindow", "Ctrl+P"))
        self.actionBunch_Length.setText(_translate("MainWindow", "Bunch Length..."))
        self.actionLongitudinal_Phase_Space.setText(_translate("MainWindow", "Longitudinal Phase Space..."))
        self.actionEmittance.setText(_translate("MainWindow", "Emittance..."))
        self.actionEmail_Maintainer.setText(_translate("MainWindow", "Email Maintainer"))
from esme.gui.sbunchpanel import SpecialBunchMidLayerPanel
from esme.gui.screen import ScreenDisplayWidget
from esme.gui.status import LPSStateWatcher
from esme.gui.tds import TDSControl


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
