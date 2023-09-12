"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QDesktopWidget,
    QTabWidget,
)
from ..tongue.tabs.segmentation import SegmentationTab
from ..tongue.tabs.reconstruction import ReconstructionTab
from ..tongue.tabs.analysis import AnalysisTab

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
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../mouse.png"
        )
        app_icon = QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setGeometry(55, 5, 1470, 800)
        self.setWindowTitle("Facemap")

        dark_stylesheet = """
            QMainWindow {background: 'black';}
            QTabWidget::pane {background-color: rgb(0,0,0);}
            QTabWidget::tab-bar {
                alignment: center;
                background-color: rgb(0, 0, 0);
                border-bottom: 1px solid rgb(30, 30, 30);
            }
            QTabBar::tab {
                height: 30px;
                width: 200px;
                font-size: 12pt;
                color: rgb(220, 220, 220);
                background-color: rgb(0, 0, 0);
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


def run():
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mouse.png"
    )
    app_icon = QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    window = MainWindow()
    return window