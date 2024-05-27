# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'special_bunch_panel.ui'
#
# Created by: PyQt5 UI code generator 5.15.10
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtWidgets


class Ui_special_bunch_panel(object):
    def setupUi(self, special_bunch_panel):
        special_bunch_panel.setObjectName("special_bunch_panel")
        special_bunch_panel.resize(460, 126)
        special_bunch_panel.setMinimumSize(QtCore.QSize(0, 100))
        special_bunch_panel.setMaximumSize(QtCore.QSize(16777215, 190))
        self.gridLayout = QtWidgets.QGridLayout(special_bunch_panel)
        self.gridLayout.setObjectName("gridLayout")
        self.go_to_last_bunch_in_br_pushbutton = QtWidgets.QPushButton(
            special_bunch_panel
        )
        self.go_to_last_bunch_in_br_pushbutton.setEnabled(True)
        self.go_to_last_bunch_in_br_pushbutton.setAutoExclusive(False)
        self.go_to_last_bunch_in_br_pushbutton.setObjectName(
            "go_to_last_bunch_in_br_pushbutton"
        )
        self.gridLayout.addWidget(self.go_to_last_bunch_in_br_pushbutton, 0, 0, 1, 1)
        self.beam_region_label = QtWidgets.QLabel(special_bunch_panel)
        self.beam_region_label.setObjectName("beam_region_label")
        self.gridLayout.addWidget(self.beam_region_label, 0, 1, 1, 1)
        self.beamregion_spinbox = QtWidgets.QSpinBox(special_bunch_panel)
        self.beamregion_spinbox.setMinimum(1)
        self.beamregion_spinbox.setObjectName("beamregion_spinbox")
        self.gridLayout.addWidget(self.beamregion_spinbox, 0, 2, 1, 1)
        self.use_fast_kickers_checkbox = QtWidgets.QCheckBox(special_bunch_panel)
        self.use_fast_kickers_checkbox.setObjectName("use_fast_kickers_checkbox")
        self.gridLayout.addWidget(self.use_fast_kickers_checkbox, 0, 3, 1, 1)
        self.go_to_last_laserpulse_pushbutton = QtWidgets.QPushButton(
            special_bunch_panel
        )
        self.go_to_last_laserpulse_pushbutton.setEnabled(True)
        self.go_to_last_laserpulse_pushbutton.setChecked(False)
        self.go_to_last_laserpulse_pushbutton.setObjectName(
            "go_to_last_laserpulse_pushbutton"
        )
        self.gridLayout.addWidget(self.go_to_last_laserpulse_pushbutton, 1, 0, 1, 1)
        self.bunch_label = QtWidgets.QLabel(special_bunch_panel)
        self.bunch_label.setObjectName("bunch_label")
        self.gridLayout.addWidget(self.bunch_label, 1, 1, 1, 1)
        self.bunch_spinbox = QtWidgets.QSpinBox(special_bunch_panel)
        self.bunch_spinbox.setMinimum(1)
        self.bunch_spinbox.setMaximum(4000)
        self.bunch_spinbox.setObjectName("bunch_spinbox")
        self.gridLayout.addWidget(self.bunch_spinbox, 1, 2, 1, 1)
        self.use_tds_checkbox = QtWidgets.QCheckBox(special_bunch_panel)
        self.use_tds_checkbox.setObjectName("use_tds_checkbox")
        self.gridLayout.addWidget(self.use_tds_checkbox, 1, 3, 1, 1)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.start_button = QtWidgets.QPushButton(special_bunch_panel)
        self.start_button.setStyleSheet("QPushButton { color: green; }")
        self.start_button.setCheckable(True)
        self.start_button.setChecked(False)
        self.start_button.setAutoRepeat(False)
        self.start_button.setObjectName("start_button")
        self.horizontalLayout_2.addWidget(self.start_button)
        self.stop_button = QtWidgets.QPushButton(special_bunch_panel)
        self.stop_button.setStyleSheet("QPushButton { color: red; }")
        self.stop_button.setCheckable(True)
        self.stop_button.setChecked(False)
        self.stop_button.setAutoRepeat(False)
        self.stop_button.setObjectName("stop_button")
        self.horizontalLayout_2.addWidget(self.stop_button)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.npulses_label = QtWidgets.QLabel(special_bunch_panel)
        self.npulses_label.setObjectName("npulses_label")
        self.gridLayout.addWidget(self.npulses_label, 2, 1, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.npulses_spinbox = QtWidgets.QSpinBox(special_bunch_panel)
        self.npulses_spinbox.setMaximum(10000000)
        self.npulses_spinbox.setProperty("value", 10000)
        self.npulses_spinbox.setObjectName("npulses_spinbox")
        self.horizontalLayout.addWidget(self.npulses_spinbox)
        self.gridLayout.addLayout(self.horizontalLayout, 2, 2, 1, 1)
        self.ibfb_checkbox = QtWidgets.QCheckBox(special_bunch_panel)
        self.ibfb_checkbox.setObjectName("ibfb_checkbox")
        self.gridLayout.addWidget(self.ibfb_checkbox, 2, 3, 1, 1)

        self.retranslateUi(special_bunch_panel)
        QtCore.QMetaObject.connectSlotsByName(special_bunch_panel)

    def retranslateUi(self, special_bunch_panel):
        _translate = QtCore.QCoreApplication.translate
        special_bunch_panel.setWindowTitle(_translate("special_bunch_panel", "Form"))
        self.go_to_last_bunch_in_br_pushbutton.setToolTip(
            _translate(
                "special_bunch_panel",
                "Set the bunch number to the last bunch in the selected beam region",
            )
        )
        self.go_to_last_bunch_in_br_pushbutton.setText(
            _translate("special_bunch_panel", "Last in Beam Region")
        )
        self.beam_region_label.setText(_translate("special_bunch_panel", "Beam Region"))
        self.beamregion_spinbox.setToolTip(
            _translate(
                "special_bunch_panel",
                "The beam region in which to select a diagnostic bunch",
            )
        )
        self.use_fast_kickers_checkbox.setToolTip(
            _translate(
                "special_bunch_panel",
                "Whether to fire the fast kickers for the diagnostic bunch",
            )
        )
        self.use_fast_kickers_checkbox.setText(
            _translate("special_bunch_panel", "Fast Kickers")
        )
        self.go_to_last_laserpulse_pushbutton.setToolTip(
            _translate(
                "special_bunch_panel",
                "Append a bunch to the last beam region in the machine",
            )
        )
        self.go_to_last_laserpulse_pushbutton.setText(
            _translate("special_bunch_panel", "Append Diag. Bunch")
        )
        self.bunch_label.setText(_translate("special_bunch_panel", "Bunch"))
        self.bunch_spinbox.setToolTip(
            _translate(
                "special_bunch_panel",
                "The bunch number within the selected beam region to optionally fire the kickers or TDS for.",
            )
        )
        self.use_tds_checkbox.setToolTip(
            _translate(
                "special_bunch_panel",
                "Whether to fire the TDS for the diagnostic bunch",
            )
        )
        self.use_tds_checkbox.setText(_translate("special_bunch_panel", "TDS"))
        self.start_button.setToolTip(
            _translate(
                "special_bunch_panel",
                "<html><head/><body><p>Start diagnostic bunch optionally with fast kickers and the TDS</p></body></html>",
            )
        )
        self.start_button.setText(_translate("special_bunch_panel", "Start"))
        self.stop_button.setToolTip(
            _translate(
                "special_bunch_panel",
                "<html><head/><body><p>Start diagnostic bunch optionally with fast kickers and the TDS</p></body></html>",
            )
        )
        self.stop_button.setText(_translate("special_bunch_panel", "Stop"))
        self.npulses_label.setText(_translate("special_bunch_panel", "Pulses"))
        self.npulses_spinbox.setToolTip(
            _translate(
                "special_bunch_panel",
                "The number of times to fire the special diagnostic bunch before stopping",
            )
        )
        self.ibfb_checkbox.setToolTip(
            _translate(
                "special_bunch_panel",
                "IBFB Adaptive FF State (Should typically be off when doing measurements)",
            )
        )
        self.ibfb_checkbox.setText(_translate("special_bunch_panel", "IBFB AFF"))


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    special_bunch_panel = QtWidgets.QWidget()
    ui = Ui_special_bunch_panel()
    ui.setupUi(special_bunch_panel)
    special_bunch_panel.show()
    sys.exit(app.exec_())
