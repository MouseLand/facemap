from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
import os
from . import guiparts, io
from PyQt5.QtGui import QPixmap, QFont, QPainterPath, QPainter, QBrush
from PyQt5.QtWidgets import QAction, QLabel

def mainmenu(parent):    
    # --------------- MENU BAR --------------------------
    # run suite2p from scratch
    openFile = QAction("&Load single movie file", parent)
    openFile.setShortcut("Ctrl+L")
    openFile.triggered.connect(lambda: io.open_file(parent))
    parent.addAction(openFile)

    openFolder = QAction("Open &Folder of movies", parent)
    openFolder.setShortcut("Ctrl+F")
    openFolder.triggered.connect(lambda: io.open_folder(parent))
    parent.addAction(openFolder)

    # load processed data
    loadProc = QAction("Load &Processed data", parent)
    loadProc.setShortcut("Ctrl+P")
    loadProc.triggered.connect(lambda: io.open_proc(parent))
    parent.addAction(loadProc)

    # Set output folder
    setOutputFolder = QAction("Set &Output folder", parent)
    setOutputFolder.setShortcut("Ctrl+O")
    setOutputFolder.triggered.connect(lambda: io.save_folder(parent))
    parent.addAction(setOutputFolder)

    loadPose = QAction("Load &pose data", parent)
    loadPose.triggered.connect(lambda: io.get_pose_file(parent))
    parent.addAction(loadPose)

    # Help menu actions
    helpContent = QAction("Help Content", parent)
    helpContent.setShortcut("Ctrl+H")
    helpContent.triggered.connect(lambda: launch_user_manual(parent))
    parent.addAction(helpContent)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(openFile)
    file_menu.addAction(openFolder)
    file_menu.addAction(loadProc)
    file_menu.addAction(loadPose)
    file_menu.addAction(setOutputFolder)
    help_menu = main_menu.addMenu("&Help")
    help_menu.addAction(helpContent)

def launch_user_manual(parent):
    w = Dialog(parent)
    w.resize(640, 480)
    w.show()

class DrawWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(DrawWidget, self).__init__(*args, **kwargs)
        self.setFixedSize(630, 470)
        icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mouse.png")
        self.logo = QPixmap(icon_path).scaled(120, 90, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.logoLabel = QLabel(self) 
        self.logoLabel.setPixmap(self.logo) 
        self.logoLabel.setScaledContents(True)
        self.logoLabel.move(240,10)
        self.logoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.helpText = QtWidgets.QPlainTextEdit(self)
        self.helpText.move(10,160)
        self.helpText.insertPlainText("The motion SVDs (small ROIs / multivideo) are computed on the movie downsampled in space by the spatial downsampling input box in the GUI (default 4 pixels). Note the saturation set in this window is NOT used for any processing.")
        self.helpText.appendPlainText("\nThe motion M is defined as the abs(current_frame - previous_frame), and the average motion energy across frames is computed using a subset of frames (avgmot) (at least 1000 frames - set at line 45 in subsampledMean.m or line 183 in process.py). Then the singular vectors of the motion energy are computed on chunks of data, also from a subset of frames (15 chunks of 1000 frames each). Let F be the chunk of frames [pixels x time]. Then")
        self.helpText.appendPlainText("\nuMot = []; \nfor j = 1:nchunks \n  M = abs(diff(F,1,2)); \n   [u,~,~] = svd(M - avgmot);\n  uMot = cat(2, uMot, u);\nend\nuMot,~,~] = svd(uMot);\nuMotMask = normc(uMot(:, 1:500)); % keep 500 components")
        self.helpText.resize(580,400)
        self.helpText.setReadOnly(True)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QBrush(QtCore.Qt.black))
        painter.setPen(QtCore.Qt.NoPen)
        path = QPainterPath()
        path.addText(QtCore.QPoint(235, 130), QFont("Times", 30, QFont.Bold), "Facemap")
        help_text = "Help content"
        path.addText(QtCore.QPoint(10, 150), QFont("Times", 20), help_text)
        painter.drawPath(path)


class Dialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super(Dialog, self).__init__(parent)
        scroll_area = QtWidgets.QScrollArea(widgetResizable=True)
        draw_widget = DrawWidget()
        scroll_area.setWidget(draw_widget)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(scroll_area)
