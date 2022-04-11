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
    openFile.setShortcut("Ctrl+O")
    openFile.triggered.connect(lambda: io.open_file(parent))
    parent.addAction(openFile)

    openFolder = QAction("Open folder of movies", parent)
    # openFolder.setShortcut("Ctrl+I")
    openFolder.triggered.connect(lambda: io.open_folder(parent))
    parent.addAction(openFolder)

    # load processed data
    loadProc = QAction("Load processed data", parent)
    loadProc.triggered.connect(lambda: io.open_proc(parent))
    parent.addAction(loadProc)

    # Set output folder
    setOutputFolder = QAction("&Set output folder", parent)
    setOutputFolder.setShortcut("Ctrl+S")
    setOutputFolder.triggered.connect(lambda: io.save_folder(parent))
    parent.addAction(setOutputFolder)

    loadPose = QAction("Load keypoints", parent)
    # loadPose.setShortcut("Ctrl+P")
    loadPose.triggered.connect(lambda: io.get_pose_file(parent))
    parent.addAction(loadPose)

    train_model = QAction("Train model", parent)
    train_model.triggered.connect(lambda: parent.show_model_training_popup())
    parent.addAction(train_model)

    load_finetuned_model = QAction("Load finetuned model", parent)
    load_finetuned_model.triggered.connect(lambda: parent.load_finetuned_model())
    parent.addAction(load_finetuned_model)

    # Help menu actions
    helpContent = QAction("Help Content", parent)
    helpContent.setShortcut("Ctrl+H")
    helpContent.triggered.connect(lambda: launch_user_manual(parent))
    parent.addAction(helpContent)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("File")
    file_menu.addAction(openFile)
    file_menu.addAction(openFolder)
    file_menu.addAction(loadProc)
    file_menu.addAction(setOutputFolder)
    pose_menu = main_menu.addMenu("Pose")
    pose_menu.addAction(loadPose)
    pose_menu.addAction(load_finetuned_model)
    pose_menu.addAction(train_model)
    help_menu = main_menu.addMenu("Help")
    help_menu.addAction(helpContent)


def launch_user_manual(parent):
    w = Dialog(parent)
    w.resize(640, 480)
    w.show()


class DrawWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(DrawWidget, self).__init__(*args, **kwargs)
        self.setFixedSize(630, 470)
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "mouse.png"
        )
        self.logo = QPixmap(icon_path).scaled(
            120, 90, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation
        )
        self.logoLabel = QLabel(self)
        self.logoLabel.setPixmap(self.logo)
        self.logoLabel.setScaledContents(True)
        self.logoLabel.move(240, 10)
        self.logoLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.helpText = QtWidgets.QPlainTextEdit(self)
        self.helpText.move(10, 160)
        self.helpText.resize(580, 400)
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
