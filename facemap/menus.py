from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from . import guiparts, io

def mainmenu(parent):
    main_menu = parent.menuBar()
    
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

    # make mainmenu!
    main_menu = parent.menuBar()
    file_menu = main_menu.addMenu("&File")
    file_menu.addAction(openFile)
    file_menu.addAction(openFolder)
    file_menu.addAction(loadProc)
    
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

