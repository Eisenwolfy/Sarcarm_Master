from PyQt5.QtWidgets import QApplication
import sys
from front import interface

if __name__ == '__main__':
    print("App is running, wait for some time...")
    app = QApplication(sys.argv)
    ex = interface.GifBackgroundApp()
    ex.show()
    sys.exit(app.exec_())
