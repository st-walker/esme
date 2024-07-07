# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'current.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_CurrentProfilerWindow(object):
    def setupUi(self, CurrentProfilerWindow):
        CurrentProfilerWindow.setObjectName("CurrentProfilerWindow")
        CurrentProfilerWindow.resize(802, 708)
        self.centralwidget = QtWidgets.QWidget(CurrentProfilerWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.area_control = AreaControl(self.centralwidget)
        self.area_control.setObjectName("area_control")
        self.verticalLayout.addWidget(self.area_control)
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.gridLayout = QtWidgets.QGridLayout(self.groupBox_2)
        self.gridLayout.setObjectName("gridLayout")
        self.author_line_edit = QtWidgets.QLineEdit(self.groupBox_2)
        self.author_line_edit.setTabletTracking(False)
        self.author_line_edit.setObjectName("author_line_edit")
        self.gridLayout.addWidget(self.author_line_edit, 0, 1, 1, 1)
        self.logbook_text_edit = QtWidgets.QTextEdit(self.groupBox_2)
        self.logbook_text_edit.setTabChangesFocus(True)
        self.logbook_text_edit.setAcceptRichText(False)
        self.logbook_text_edit.setObjectName("logbook_text_edit")
        self.gridLayout.addWidget(self.logbook_text_edit, 1, 0, 1, 2)
        self.author_label = QtWidgets.QLabel(self.groupBox_2)
        self.author_label.setObjectName("author_label")
        self.gridLayout.addWidget(self.author_label, 0, 0, 1, 1)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.progress_bar = QtWidgets.QProgressBar(self.groupBox)
        self.progress_bar.setProperty("value", 0)
        self.progress_bar.setObjectName("progress_bar")
        self.gridLayout_2.addWidget(self.progress_bar, 6, 0, 1, 4)
        self.line = QtWidgets.QFrame(self.groupBox)
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_2.addWidget(self.line, 2, 0, 1, 4)
        self.start_measurement_button = QtWidgets.QPushButton(self.groupBox)
        self.start_measurement_button.setObjectName("start_measurement_button")
        self.gridLayout_2.addWidget(self.start_measurement_button, 7, 0, 1, 2)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.groupBox)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.bunch_number_spinner = QtWidgets.QSpinBox(self.groupBox)
        self.bunch_number_spinner.setMinimum(1)
        self.bunch_number_spinner.setObjectName("bunch_number_spinner")
        self.horizontalLayout_2.addWidget(self.bunch_number_spinner)
        self.gridLayout_2.addLayout(self.horizontalLayout_2, 0, 2, 1, 2)
        self.cancel_button = QtWidgets.QPushButton(self.groupBox)
        self.cancel_button.setEnabled(False)
        self.cancel_button.setObjectName("cancel_button")
        self.gridLayout_2.addWidget(self.cancel_button, 7, 2, 1, 2)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.groupBox)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.beam_region_spinner = QtWidgets.QSpinBox(self.groupBox)
        self.beam_region_spinner.setMinimum(1)
        self.beam_region_spinner.setObjectName("beam_region_spinner")
        self.horizontalLayout.addWidget(self.beam_region_spinner)
        self.gridLayout_2.addLayout(self.horizontalLayout, 0, 0, 1, 2)
        self.goto_last_in_beamregion = QtWidgets.QPushButton(self.groupBox)
        self.goto_last_in_beamregion.setObjectName("goto_last_in_beamregion")
        self.gridLayout_2.addWidget(self.goto_last_in_beamregion, 1, 0, 1, 4)
        self.verticalLayout.addWidget(self.groupBox)
        self.gridLayout_3.addLayout(self.verticalLayout, 0, 0, 3, 1)
        self.current_graphics = GraphicsLayoutWidget(self.centralwidget)
        self.current_graphics.setObjectName("current_graphics")
        self.gridLayout_3.addWidget(self.current_graphics, 0, 1, 1, 1)
        self.tilt_graphics = GraphicsLayoutWidget(self.centralwidget)
        self.tilt_graphics.setObjectName("tilt_graphics")
        self.gridLayout_3.addWidget(self.tilt_graphics, 1, 1, 1, 1)
        self.beam_table_view = BeamCurrentTableView(self.centralwidget)
        self.beam_table_view.setObjectName("beam_table_view")
        self.gridLayout_3.addWidget(self.beam_table_view, 2, 1, 1, 1)
        CurrentProfilerWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(CurrentProfilerWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 802, 24))
        self.menubar.setObjectName("menubar")
        CurrentProfilerWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(CurrentProfilerWindow)
        self.statusbar.setObjectName("statusbar")
        CurrentProfilerWindow.setStatusBar(self.statusbar)

        self.retranslateUi(CurrentProfilerWindow)
        QtCore.QMetaObject.connectSlotsByName(CurrentProfilerWindow)

    def retranslateUi(self, CurrentProfilerWindow):
        _translate = QtCore.QCoreApplication.translate
        CurrentProfilerWindow.setWindowTitle(
            _translate("CurrentProfilerWindow", "MainWindow")
        )
        self.groupBox_2.setTitle(_translate("CurrentProfilerWindow", "e-LogBook"))
        self.author_line_edit.setPlaceholderText(
            _translate("CurrentProfilerWindow", "xfeloper")
        )
        self.logbook_text_edit.setPlaceholderText(
            _translate("CurrentProfilerWindow", "Logbook entry...")
        )
        self.author_label.setText(_translate("CurrentProfilerWindow", "Author"))
        self.groupBox.setTitle(_translate("CurrentProfilerWindow", "Measurement"))
        self.start_measurement_button.setText(
            _translate("CurrentProfilerWindow", "Measure Current Profile")
        )
        self.label_2.setText(_translate("CurrentProfilerWindow", "Bunch Number"))
        self.cancel_button.setText(_translate("CurrentProfilerWindow", "Cancel"))
        self.label.setText(_translate("CurrentProfilerWindow", "Beam Region"))
        self.goto_last_in_beamregion.setText(
            _translate("CurrentProfilerWindow", "Go to last bunch in Beam Region")
        )


from pyqtgraph import GraphicsLayoutWidget

from esme.gui.widgets.area import AreaControl
from esme.gui.widgets.table import BeamCurrentTableView

if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    CurrentProfilerWindow = QtWidgets.QMainWindow()
    ui = Ui_CurrentProfilerWindow()
    ui.setupUi(CurrentProfilerWindow)
    CurrentProfilerWindow.show()
    sys.exit(app.exec_())