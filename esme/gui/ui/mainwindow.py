# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'mainwindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_tan1_mainwindow(object):
    def setupUi(self, tan1_mainwindow):
        tan1_mainwindow.setObjectName("tan1_mainwindow")
        tan1_mainwindow.resize(1400, 785)
        tan1_mainwindow.setDocumentMode(False)
        self.centralwidget = QtWidgets.QWidget(tan1_mainwindow)
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
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.controls_group_box = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.controls_group_box.sizePolicy().hasHeightForWidth()
        )
        self.controls_group_box.setSizePolicy(sizePolicy)
        self.controls_group_box.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.controls_group_box.setObjectName("controls_group_box")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.controls_group_box)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.area = AreaControl(self.controls_group_box)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.area.sizePolicy().hasHeightForWidth())
        self.area.setSizePolicy(sizePolicy)
        self.area.setObjectName("area")
        self.gridLayout_2.addWidget(self.area, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.controls_group_box)
        self.groupBox_3 = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.groupBox_3.sizePolicy().hasHeightForWidth())
        self.groupBox_3.setSizePolicy(sizePolicy)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.groupBox_3)
        self.verticalLayout.setContentsMargins(-1, -1, -1, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.special_bunch_panel = SpecialBunchMidLayerPanel(self.groupBox_3)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.special_bunch_panel.sizePolicy().hasHeightForWidth()
        )
        self.special_bunch_panel.setSizePolicy(sizePolicy)
        self.special_bunch_panel.setObjectName("special_bunch_panel")
        self.verticalLayout.addWidget(self.special_bunch_panel)
        self.verticalLayout_4.addWidget(self.groupBox_3)
        self.groupBox_2 = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Minimum
        )
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(1)
        sizePolicy.setHeightForWidth(self.groupBox_2.sizePolicy().hasHeightForWidth())
        self.groupBox_2.setSizePolicy(sizePolicy)
        self.groupBox_2.setCheckable(False)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.tds_panel = TDSControl(self.groupBox_2)
        self.tds_panel.setObjectName("tds_panel")
        self.gridLayout_4.addWidget(self.tds_panel, 0, 0, 1, 1)
        self.verticalLayout_4.addWidget(self.groupBox_2)
        self.groupBox_4 = QtWidgets.QGroupBox(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
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
        self.verticalLayout_4.addWidget(self.groupBox_4)
        self.gridLayout_3.addLayout(self.verticalLayout_4, 0, 1, 1, 1)
        self.imaging_widget = ImagingControlWidget(self.maintab)
        sizePolicy = QtWidgets.QSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred
        )
        sizePolicy.setHorizontalStretch(1)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(
            self.imaging_widget.sizePolicy().hasHeightForWidth()
        )
        self.imaging_widget.setSizePolicy(sizePolicy)
        self.imaging_widget.setObjectName("imaging_widget")
        self.gridLayout_3.addWidget(self.imaging_widget, 0, 0, 1, 1)
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
        tan1_mainwindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(tan1_mainwindow)
        self.statusbar.setObjectName("statusbar")
        tan1_mainwindow.setStatusBar(self.statusbar)
        self.menubar = QtWidgets.QMenuBar(tan1_mainwindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1400, 24))
        self.menubar.setNativeMenuBar(False)
        self.menubar.setObjectName("menubar")
        self.menuMenu = QtWidgets.QMenu(self.menubar)
        self.menuMenu.setObjectName("menuMenu")
        self.menuOptics = QtWidgets.QMenu(self.menubar)
        self.menuOptics.setObjectName("menuOptics")
        tan1_mainwindow.setMenuBar(self.menubar)
        self.action_close = QtWidgets.QAction(tan1_mainwindow)
        self.action_close.setObjectName("action_close")
        self.action_print_to_logbook = QtWidgets.QAction(tan1_mainwindow)
        self.action_print_to_logbook.setObjectName("action_print_to_logbook")
        self.actionBunch_Length = QtWidgets.QAction(tan1_mainwindow)
        self.actionBunch_Length.setObjectName("actionBunch_Length")
        self.actionLongitudinal_Phase_Space = QtWidgets.QAction(tan1_mainwindow)
        self.actionLongitudinal_Phase_Space.setObjectName(
            "actionLongitudinal_Phase_Space"
        )
        self.actionEmittance = QtWidgets.QAction(tan1_mainwindow)
        self.actionEmittance.setObjectName("actionEmittance")
        self.actionEmail_Maintainer = QtWidgets.QAction(tan1_mainwindow)
        self.actionEmail_Maintainer.setObjectName("actionEmail_Maintainer")
        self.menuMenu.addAction(self.action_print_to_logbook)
        self.menuMenu.addAction(self.action_close)
        self.menubar.addAction(self.menuMenu.menuAction())
        self.menubar.addAction(self.menuOptics.menuAction())

        self.retranslateUi(tan1_mainwindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(tan1_mainwindow)

    def retranslateUi(self, tan1_mainwindow):
        _translate = QtCore.QCoreApplication.translate
        tan1_mainwindow.setWindowTitle(
            _translate("tan1_mainwindow", "Diagnostic Sections Utility")
        )
        self.controls_group_box.setTitle(
            _translate("tan1_mainwindow", "Diagnostic Area")
        )
        self.groupBox_3.setTitle(
            _translate("tan1_mainwindow", "Special Bunch Midlayer")
        )
        self.groupBox_2.setTitle(
            _translate("tan1_mainwindow", "Transverse Deflecting Structure")
        )
        self.groupBox_4.setTitle(_translate("tan1_mainwindow", "Machine State"))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.maintab), _translate("tan1_mainwindow", "Main")
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.logtab), _translate("tan1_mainwindow", "Log")
        )
        self.menuMenu.setTitle(_translate("tan1_mainwindow", "Menu"))
        self.menuOptics.setTitle(_translate("tan1_mainwindow", "Optics"))
        self.action_close.setText(_translate("tan1_mainwindow", "Close"))
        self.action_close.setShortcut(_translate("tan1_mainwindow", "Ctrl+Q"))
        self.action_print_to_logbook.setText(
            _translate("tan1_mainwindow", "Print to Logbook")
        )
        self.action_print_to_logbook.setShortcut(
            _translate("tan1_mainwindow", "Ctrl+P")
        )
        self.actionBunch_Length.setText(
            _translate("tan1_mainwindow", "Bunch Length...")
        )
        self.actionLongitudinal_Phase_Space.setText(
            _translate("tan1_mainwindow", "Longitudinal Phase Space...")
        )
        self.actionEmittance.setText(_translate("tan1_mainwindow", "Emittance..."))
        self.actionEmail_Maintainer.setText(
            _translate("tan1_mainwindow", "Email Maintainer")
        )


from esme.gui.widgets.area import AreaControl
from esme.gui.widgets.imaging import ImagingControlWidget
from esme.gui.widgets.sbunchpanel import SpecialBunchMidLayerPanel
from esme.gui.widgets.status import LPSStateWatcher
from esme.gui.widgets.tds import TDSControl

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    tan1_mainwindow = QtWidgets.QMainWindow()
    ui = Ui_tan1_mainwindow()
    ui.setupUi(tan1_mainwindow)
    tan1_mainwindow.show()
    sys.exit(app.exec_())
