# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.7
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_tdsfriend_mainwindow(object):
    def setupUi(self, tdsfriend_mainwindow):
        tdsfriend_mainwindow.setObjectName("tdsfriend_mainwindow")
        tdsfriend_mainwindow.resize(2100, 1338)
        tdsfriend_mainwindow.setLocale(QtCore.QLocale(QtCore.QLocale.German, QtCore.QLocale.Germany))
        tdsfriend_mainwindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(tdsfriend_mainwindow)
        self.centralwidget.setAutoFillBackground(False)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_9 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_9.setObjectName("gridLayout_9")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setTabBarAutoHide(True)
        self.tabWidget.setObjectName("tabWidget")
        self.maintab = QtWidgets.QWidget()
        self.maintab.setObjectName("maintab")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.maintab)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.beam_on_off_button = QtWidgets.QPushButton(self.maintab)
        self.beam_on_off_button.setObjectName("beam_on_off_button")
        self.gridLayout_5.addWidget(self.beam_on_off_button, 0, 1, 1, 2)
        self.imaging_widget = ImagingControlWidget(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.imaging_widget.sizePolicy().hasHeightForWidth())
        self.imaging_widget.setSizePolicy(sizePolicy)
        self.imaging_widget.setObjectName("imaging_widget")
        self.gridLayout_5.addWidget(self.imaging_widget, 0, 0, 2, 1)
        self.verticalLayout_3 = QtWidgets.QVBoxLayout()
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.controls_group_box = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.controls_group_box.sizePolicy().hasHeightForWidth())
        self.controls_group_box.setSizePolicy(sizePolicy)
        self.controls_group_box.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.controls_group_box.setObjectName("controls_group_box")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.controls_group_box)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.area = AreaControl(self.controls_group_box)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.area.sizePolicy().hasHeightForWidth())
        self.area.setSizePolicy(sizePolicy)
        self.area.setObjectName("area")
        self.gridLayout_2.addWidget(self.area, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.controls_group_box)
        self.groupBox_3 = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.special_bunch_panel = SpecialBunchMidLayerPanel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.special_bunch_panel.sizePolicy().hasHeightForWidth())
        self.special_bunch_panel.setSizePolicy(sizePolicy)
        self.special_bunch_panel.setObjectName("special_bunch_panel")
        self.verticalLayout.addWidget(self.special_bunch_panel)
        self.verticalLayout_3.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setCheckable(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tds_panel = TDSControl(self.groupBox_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.tds_panel.sizePolicy().hasHeightForWidth())
        self.tds_panel.setSizePolicy(sizePolicy)
        self.tds_panel.setObjectName("tds_panel")
        self.gridLayout_4.addWidget(self.tds_panel, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.maintab)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.target_stack = TargetStack(self.groupBox)
        self.target_stack.setObjectName("target_stack")
        self.page = QtWidgets.QWidget()
        self.page.setObjectName("page")
        self.target_stack.addWidget(self.page)
        self.page_2 = QtWidgets.QWidget()
        self.page_2.setObjectName("page_2")
        self.target_stack.addWidget(self.page_2)
        self.gridLayout_3.addWidget(self.target_stack, 0, 0, 1, 1)
        self.verticalLayout_3.addWidget(self.groupBox)
        self.groupBox_4 = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_4.sizePolicy().hasHeightForWidth())
        self.groupBox_4.setSizePolicy(sizePolicy)
        self.groupBox_4.setObjectName("groupBox_4")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.groupBox_4)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.machine_state_widget = LPSStateWatcher(self.groupBox_4)
        self.machine_state_widget.setObjectName("machine_state_widget")
        self.verticalLayout_2.addWidget(self.machine_state_widget)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.gridLayout_5.addLayout(self.verticalLayout_3, 1, 1, 1, 2)
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
        tdsfriend_mainwindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(tdsfriend_mainwindow)
        self.statusbar.setObjectName("statusbar")
        tdsfriend_mainwindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(tdsfriend_mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2100, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menu_external = QtWidgets.QMenu(self.menubar)
        self.menu_external.setObjectName("menu_external")
        self.menuExpert = QtWidgets.QMenu(self.menu_external)
        self.menuExpert.setObjectName("menuExpert")
        self.menu_i1 = QtWidgets.QMenu(self.menuExpert)
        self.menu_i1.setObjectName("menu_i1")
        self.menu_b2 = QtWidgets.QMenu(self.menuExpert)
        self.menu_b2.setObjectName("menu_b2")
        tdsfriend_mainwindow.setMenuBar(self.menubar)
        self.action_close = QtWidgets.QAction(tdsfriend_mainwindow)
        self.action_close.setObjectName("action_close")
        self.action_print_to_logbook = QtWidgets.QAction(tdsfriend_mainwindow)
        self.action_print_to_logbook.setObjectName("action_print_to_logbook")
        self.action_pattern_builder = QtWidgets.QAction(tdsfriend_mainwindow)
        self.action_pattern_builder.setObjectName("action_pattern_builder")
        self.action_camera_status = QtWidgets.QAction(tdsfriend_mainwindow)
        self.action_camera_status.setObjectName("action_camera_status")
        self.action_image_analysis_server = QtWidgets.QAction(tdsfriend_mainwindow)
        self.action_image_analysis_server.setObjectName("action_image_analysis_server")
        self.actionSpecial_Bunch_Midlayer_i1 = QtWidgets.QAction(tdsfriend_mainwindow)
        self.actionSpecial_Bunch_Midlayer_i1.setObjectName("actionSpecial_Bunch_Midlayer_i1")
        self.actionLLRF_b2 = QtWidgets.QAction(tdsfriend_mainwindow)
        self.actionLLRF_b2.setObjectName("actionLLRF_b2")
        self.actionSpecial_Bunch_Midlayer_b2 = QtWidgets.QAction(tdsfriend_mainwindow)
        self.actionSpecial_Bunch_Midlayer_b2.setObjectName("actionSpecial_Bunch_Midlayer_b2")
        self.actionLLRF_i1 = QtWidgets.QAction(tdsfriend_mainwindow)
        self.actionLLRF_i1.setObjectName("actionLLRF_i1")
        self.actionBLM_Toroid_Alarm_Overview = QtWidgets.QAction(tdsfriend_mainwindow)
        self.actionBLM_Toroid_Alarm_Overview.setObjectName("actionBLM_Toroid_Alarm_Overview")
        self.menuMenu.addAction(self.action_print_to_logbook)
        self.menuMenu.addAction(self.action_close)
        self.menu_i1.addAction(self.actionLLRF_i1)
        self.menu_i1.addAction(self.actionSpecial_Bunch_Midlayer_i1)
        self.menu_b2.addAction(self.actionLLRF_b2)
        self.menu_b2.addAction(self.actionSpecial_Bunch_Midlayer_b2)
        self.menuExpert.addAction(self.action_image_analysis_server)
        self.menuExpert.addAction(self.menu_i1.menuAction())
        self.menuExpert.addAction(self.menu_b2.menuAction())
        self.menu_external.addAction(self.action_pattern_builder)
        self.menu_external.addAction(self.action_camera_status)
        self.menu_external.addAction(self.menuExpert.menuAction())
        self.menu_external.addAction(self.actionBLM_Toroid_Alarm_Overview)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menu_external.menuAction())

        self.retranslateUi(tdsfriend_mainwindow)
        self.tabWidget.setCurrentIndex(0)
        self.target_stack.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(tdsfriend_mainwindow)

    def retranslateUi(self, tdsfriend_mainwindow):
        _translate = QtCore.QCoreApplication.translate
        tdsfriend_mainwindow.setWindowTitle(_translate("tdsfriend_mainwindow", "Diagnostic Sections Utility"))
        self.beam_on_off_button.setText(_translate("tdsfriend_mainwindow", "Beam On"))
        self.controls_group_box.setTitle(_translate("tdsfriend_mainwindow", "Diagnostic Section"))
        self.groupBox_3.setTitle(_translate("tdsfriend_mainwindow", "Special Bunch Midlayer"))
        self.groupBox_2.setTitle(_translate("tdsfriend_mainwindow", "Transverse Deflecting Structure"))
        self.groupBox.setTitle(_translate("tdsfriend_mainwindow", "Target"))
        self.groupBox_4.setTitle(_translate("tdsfriend_mainwindow", "Machine State"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.maintab), _translate("tdsfriend_mainwindow", "Main"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.logtab), _translate("tdsfriend_mainwindow", "Log"))
        self.menuMenu.setTitle(_translate("tdsfriend_mainwindow", "Menu"))
        self.menu_external.setTitle(_translate("tdsfriend_mainwindow", "External"))
        self.menuExpert.setTitle(_translate("tdsfriend_mainwindow", "Expert"))
        self.menu_i1.setTitle(_translate("tdsfriend_mainwindow", "I1"))
        self.menu_b2.setTitle(_translate("tdsfriend_mainwindow", "B2"))
        self.action_close.setText(_translate("tdsfriend_mainwindow", "Quit TDSChum"))
        self.action_close.setShortcut(_translate("tdsfriend_mainwindow", "Ctrl+Q"))
        self.action_print_to_logbook.setText(_translate("tdsfriend_mainwindow", "Print to Logbook"))
        self.action_print_to_logbook.setToolTip(_translate("tdsfriend_mainwindow", "Send Screenshot of Window to e-LogBook"))
        self.action_print_to_logbook.setShortcut(_translate("tdsfriend_mainwindow", "Ctrl+P"))
        self.action_pattern_builder.setText(_translate("tdsfriend_mainwindow", "Pattern Builder"))
        self.action_pattern_builder.setToolTip(_translate("tdsfriend_mainwindow", "Open Pattern Builder"))
        self.action_camera_status.setText(_translate("tdsfriend_mainwindow", "Camera Status"))
        self.action_camera_status.setToolTip(_translate("tdsfriend_mainwindow", "Open Camera Status Window"))
        self.action_image_analysis_server.setText(_translate("tdsfriend_mainwindow", "Image Analysis Server"))
        self.action_image_analysis_server.setToolTip(_translate("tdsfriend_mainwindow", "Open the Image Analysis Server Expert Panel"))
        self.actionSpecial_Bunch_Midlayer_i1.setText(_translate("tdsfriend_mainwindow", "Special Bunch Midlayer"))
        self.actionSpecial_Bunch_Midlayer_i1.setToolTip(_translate("tdsfriend_mainwindow", "Open Special Bunch Midlayer Panel for I1"))
        self.actionLLRF_b2.setText(_translate("tdsfriend_mainwindow", "LLRF"))
        self.actionLLRF_b2.setToolTip(_translate("tdsfriend_mainwindow", "Open LLRF Panel for the B2 TDS"))
        self.actionSpecial_Bunch_Midlayer_b2.setText(_translate("tdsfriend_mainwindow", "Special Bunch Midlayer"))
        self.actionSpecial_Bunch_Midlayer_b2.setToolTip(_translate("tdsfriend_mainwindow", "Open Special Bunch Midlayer Panel for B2"))
        self.actionLLRF_i1.setText(_translate("tdsfriend_mainwindow", "LLRF"))
        self.actionLLRF_i1.setToolTip(_translate("tdsfriend_mainwindow", "Open LLRF Panel for the I1 TDS"))
        self.actionBLM_Toroid_Alarm_Overview.setText(_translate("tdsfriend_mainwindow", "BLM && Toroid Alarm Overview"))
from esme.gui.widgets.area import AreaControl
from esme.gui.widgets.imaging import ImagingControlWidget
from esme.gui.widgets.sbunchpanel import SpecialBunchMidLayerPanel
from esme.gui.widgets.status import LPSStateWatcher
from esme.gui.widgets.target import TargetStack
from esme.gui.widgets.tds import TDSControl


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    tdsfriend_mainwindow = QtWidgets.QMainWindow()
    ui = Ui_tdsfriend_mainwindow()
    ui.setupUi(tdsfriend_mainwindow)
    tdsfriend_mainwindow.show()
    sys.exit(app.exec_())
