from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from . import guiparts, io

def mainmenu(parent):    
    # --------------- MENU BAR --------------------------
    # run suite2p from scratch
    openFile = QtGui.QAction("&Load single movie file", parent)
    openFile.setShortcut("Ctrl+L")
    openFile.triggered.connect(lambda: io.open_file(parent))
    parent.addAction(openFile)

    openFolder = QtGui.QAction("Open &Folder of movies", parent)
    openFolder.setShortcut("Ctrl+F")
    openFolder.triggered.connect(lambda: io.open_folder(parent))
    parent.addAction(openFolder)

    # load processed data
    loadProc = QtGui.QAction("Load &Processed data", parent)
    loadProc.setShortcut("Ctrl+P")
    loadProc.triggered.connect(lambda: io.open_proc(parent))
    parent.addAction(loadProc)

    # Help menu actions
    helpContent = QtGui.QAction("Help Content", parent)
    helpContent.setShortcut("Ctrl+H")
    helpContent.triggered.connect(lambda: launch_user_manual(parent))
    parent.addAction(helpContent)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(openFile)
    file_menu.addAction(openFolder)
    file_menu.addAction(loadProc)
    help_menu = main_menu.addMenu("&Help")
    help_menu.addAction(helpContent)

def launch_user_manual(parent):
    w = Dialog(parent)
    w.resize(640, 480)
    w.show()

class DrawWidget(QtWidgets.QWidget):
    def __init__(self, *args, **kwargs):
        super(DrawWidget, self).__init__(*args, **kwargs)
        self.setFixedSize(640, 480)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setBrush(QtGui.QBrush(QtCore.Qt.black))
        painter.setPen(QtCore.Qt.NoPen)
        path = QtGui.QPainterPath()
        path.addText(QtCore.QPoint(10, 50), QtGui.QFont("Times", 40, QtGui.QFont.Bold), "Facemap")
        help_text = "Add help content here"
        path.addText(QtCore.QPoint(10, 80), QtGui.QFont("Times", 14), help_text)
        painter.drawPath(path)

class Dialog(QtWidgets.QDialog):
    def __init__(self, parent):
        super(Dialog, self).__init__(parent)
        scroll_area = QtWidgets.QScrollArea(widgetResizable=True)
        draw_widget = DrawWidget()
        scroll_area.setWidget(draw_widget)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(scroll_area)
