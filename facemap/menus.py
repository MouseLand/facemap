from PyQt5 import QtGui, QtCore
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

    about = QtGui.QAction("&About", parent)
    about.triggered.connect(lambda: launch_about(parent))
    parent.addAction(about)

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(openFile)
    file_menu.addAction(openFolder)
    file_menu.addAction(loadProc)
    help_menu = main_menu.addMenu("&Help")
    help_menu.addAction(helpContent)
    help_menu.addAction(about)

def launch_about(parent):
    widget = QtGui.QDialog(parent)
    ui=Ui_Help()
    ui.setupUi(widget)
    widget.exec_()

def launch_user_manual(parent):
    widget = QtGui.QDialog(parent)
    ui=Ui_Help()
    ui.setupUi(widget)
    widget.exec_()
    
class Ui_Help(object):
    def setupUi(self, Help):
        Help.setObjectName("Help")
        Help.resize(400, 200)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("logo.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        Help.setWindowIcon(icon)
        self.gridLayoutWidget = QtGui.QWidget(Help)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(9, 9, 231, 81))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtGui.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")

        QtCore.QMetaObject.connectSlotsByName(Help)
#def onlinemenu(parent):
#    # make mainmenu!
#    main_menu = parent.menuBar()
#    online_menu = main_menu.addMenu("&Online")
#    chooseFolder = QtGui.QAction("Choose folder with frames", parent)
#    chooseFolder.setShortcut("Ctrl+O")
#    chooseFolder.triggered.connect(lambda: online.choose_folder(parent))
#    parent.addAction(chooseFolder)
#    online_menu.addAction(chooseFolder)    
#    parent.online_mode = False

