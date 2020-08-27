from PyQt5 import QtCore, QtGui, QtWidgets
import sys
from ui_my import Ui_MainWindow

import py_compile
py_compile.compile('main.py')

# создаем приложение
app = QtWidgets.QApplication(sys.argv)

# инициализация формы и интерфейс
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)
MainWindow.show()

# код
def bp():
    ui.lineEdit.setText("Hello, world!")
ui.pushButton_10.clicked.connect(bp)

# запуск
sys.exit(app.exec_())
