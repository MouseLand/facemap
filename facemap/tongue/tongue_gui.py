import os, sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtCore import Qt


import sys
from PyQt5 import QtCore, QtGui, QtWidgets, uic
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QTabWidget,
    QWidget,
    QVBoxLayout,
    QLabel,
)
class SegmentationTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("Segmentation Tab"))
        self.setLayout(self.layout)

class ReconstructionTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("3D Reconstruction Tab"))
        self.setLayout(self.layout)

class AnalysisTab(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.layout.addWidget(QLabel("Analysis Tab"))
        self.setLayout(self.layout)

class TabWidget(QTabWidget):
    def __init__(self):
        super().__init__()
        self.setTabShape(QTabWidget.Rounded)
        self.setTabPosition(QTabWidget.North)

        segmentation_icon = QIcon.fromTheme("/home/asyeda/Facemap/facemap/facemap/tongue/icons/segmentation_icon.png")
        reconstruction_icon = QIcon.fromTheme("/home/asyeda/Facemap/facemap/facemap/tongue/icons/reconstruction_icon.png")
        analysis_icon = QIcon.fromTheme("/home/asyeda/Facemap/facemap/facemap/tongue/icons/analysis_icon.png")

        segmentation_tab = SegmentationTab()
        reconstruction_tab = ReconstructionTab()
        analysis_tab = AnalysisTab()

        self.addTab(segmentation_tab, segmentation_icon, "Segmentation")
        self.addTab(reconstruction_tab, reconstruction_icon, "3D Reconstruction")
        self.addTab(analysis_tab, analysis_icon, "Analysis")

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setGeometry(55, 5, 1470, 800)
        self.setWindowTitle("Facemap")

        dark_stylesheet = """
            QMainWindow {background: 'black';}
            QTabWidget::pane {background-color: rgb(50,50,50);}
            QTabWidget::tab-bar {
                alignment: center;
                background-color: rgb(80, 80, 80);
                border-bottom: 1px solid rgb(30, 30, 30);
            }
            QTabBar::tab {
                height: 30px;
                width: 200px;
                font-size: 12pt;
                color: rgb(220, 220, 220);
                background-color: rgb(80, 80, 80);
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                padding: 5px;
            }
            QTabBar::tab:selected {
                color: rgb(220, 220, 220);
                background-color: rgb(196, 108, 57);
                font-weight: bold;
            }
            QTabBar::tab:!selected {
                color: rgb(220, 220, 220);
                background-color: rgb(50, 50, 50);
            }
            """
        self.setStyleSheet(dark_stylesheet)

        self.sizeObject = QDesktopWidget().screenGeometry(-1)
        self.resize(self.sizeObject.width(), self.sizeObject.height())

        tabs = TabWidget()
        self.setCentralWidget(tabs)


    def add_menu(self):
        self.menu = self.menuBar()
        self.file_menu = self.menu.addMenu("File")
        self.file_menu.addAction("Open")
        self.file_menu.addAction("Save")
        self.file_menu.addAction("Save As")
        self.file_menu.addAction("Quit")

    def add_buttons(self):
        # Add a button to show 
        return


    def add_canvas(self):
        self.label = QtWidgets.QLabel()
        canvas = QtGui.QPixmap(400, 300)
        canvas.fill(Qt.white)
        self.label.setPixmap(canvas)
        self.setCentralWidget(self.label)

        self.last_x, self.last_y = None, None

    def mouseMoveEvent(self, e):
        if self.last_x is None: # First event.
            self.last_x = e.x()
            self.last_y = e.y()
            return # Ignore the first time.

        painter = QtGui.QPainter(self.label.pixmap())
        painter.drawLine(self.last_x, self.last_y, e.x(), e.y())
        painter.end()
        self.update()

        # Update the origin for next time.
        self.last_x = e.x()
        self.last_y = e.y()

    def mouseReleaseEvent(self, e):
        self.last_x = None
        self.last_y = None

def run():
    # Always start by initializing Qt (only once per application)
    app = QtWidgets.QApplication(sys.argv)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "../mouse.png"
    )
    app_icon = QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    app.setApplicationName("Facemap")
    window = MainWindow()
    window.show()
    app.exec_()
