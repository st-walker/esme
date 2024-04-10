
from PyQt5 import QtWidgets



if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    conf = ScannerConfDialog()
    conf.show()
    sys.exit(app.exec_())
