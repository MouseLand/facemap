import sys, os, shutil, glob, time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
import pims
from FaceMap import facemap, roi
from scipy.stats import zscore, skew
from matplotlib import cm

istr = ['pupil', 'motSVD', 'blink', 'running']

def video_placement(Ly, Lx):
    ''' Ly and Lx are lists of video sizes '''
    npix = Ly * Lx
    picked = np.zeros((Ly.size,), np.bool)
    ly = 0
    lx = 0
    sy = np.zeros(Ly.shape, int)
    sx = np.zeros(Lx.shape, int)
    if Ly.size==2:
        gridy = 1
        gridx = 2
    elif Ly.size==3:
        gridy = 1
        gridx = 2
    else:
        gridy = int(np.round(Ly.size**0.5 * 0.75))
        gridx = int(np.ceil(Ly.size / gridy))
    LY = 0
    LX = 0
    iy = 0
    ix = 0
    while (~picked).sum() > 0:
        # place biggest movie first
        npix0 = npix.copy()
        npix0[picked] = 0
        imax = np.argmax(npix0)
        picked[imax] = 1
        if iy==0:
            ly = 0
            rowmax=0
        if ix==0:
            lx = 0
        sy[imax] = ly
        sx[imax] = lx

        ly+=Ly[imax]
        rowmax = max(rowmax, Lx[imax])
        if iy==gridy-1 or (~picked).sum()==0:
            lx+=rowmax
        LY = max(LY, ly)
        iy+=1
        if iy >= gridy:
            iy = 0
            ix += 1
    LX = lx
    return LY, LX, sy, sx

### custom QDialog which makes a list of items you can include/exclude
class ListChooser(QtGui.QDialog):
    def __init__(self, title, parent):
        super(ListChooser, self).__init__(parent)
        self.setGeometry(300,300,320,320)
        self.setWindowTitle(title)
        self.win = QtGui.QWidget(self)
        layout = QtGui.QGridLayout()
        self.win.setLayout(layout)
        #self.setCentralWidget(self.win)
        layout.addWidget(QtGui.QLabel('click to select videos (none selected => all used)'),0,0,1,1)
        self.list = QtGui.QListWidget(parent)
        for f in parent.filelist:
            self.list.addItem(f)
        layout.addWidget(self.list,1,0,7,4)
        #self.list.resize(450,250)
        self.list.setSelectionMode(QtGui.QAbstractItemView.MultiSelection)
        done = QtGui.QPushButton('done')
        done.clicked.connect(lambda: self.exit_list(parent))
        layout.addWidget(done,8,0,1,1)

    def exit_list(self, parent):
        parent.filelist = []
        items = self.list.selectedItems()
        for i in range(len(items)):
            parent.filelist.append(str(self.list.selectedItems()[i].text()))
        self.accept()

class Slider(QtGui.QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        initval = [99,99]
        self.bid = bid
        self.setOrientation(QtCore.Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(initval[bid])
        self.setTickInterval(10)
        self.valueChanged.connect(lambda: self.level_change(parent,bid))
        self.setTracking(False)

    def level_change(self, parent, bid):
        parent.sat[bid] = float(self.value())/100 * 255
        if bid==0:
            parent.pimg.setLevels([0, parent.sat[bid]])
        else:
            #parent.pROIimg.setLevels([0, parent.sat[bid]])
            parent.saturation[parent.iROI] = parent.sat[bid]
            if len(parent.ROIs) > 0:
                parent.ROIs[parent.iROI].plot(parent)
        parent.win.show()

class MainW(QtGui.QMainWindow):
    def __init__(self):
        super(MainW, self).__init__()
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "mouse.png"
        )
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(96, 96))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)

        pg.setConfigOptions(imageAxisOrder='row-major')
        self.setGeometry(15,15,1470,1000)
        self.setWindowTitle('FaceMap')
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")

        try:
            # try to load user settings
            opsfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ops_user.npy')
            self.ops = np.load(opsfile).item()
        except:
            self.ops = {'sbin': 4, 'pupil_sigma': 2., 'fullSVD': False,
                        'save_path': '', 'save_mat': False}

        self.save_path = self.ops['save_path']

        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        #layout = QtGui.QFormLayout()
        self.cwidget.setLayout(self.l0)
        #self.p0 = pg.ViewBox(lockAspect=False,name='plot1',border=[100,100,100],invertY=True)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,1,3,37,15)
        layout = self.win.ci.layout

        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True)
        #self.p0.setMouseEnabled(x=False,y=False)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.addItem(self.pimg)

        # image ROI
        self.pROI = self.win.addViewBox(lockAspect=True,row=0,col=1,invertY=True)
        #self.p0.setMouseEnabled(x=False,y=False)
        self.pROI.setMenuEnabled(False)
        self.pROIimg = pg.ImageItem()
        self.pROI.addItem(self.pROIimg)
        self.scatter = pg.ScatterPlotItem([0], [0], pen='k', symbol='+')
        self.pROI.addItem(self.scatter)

        # roi initializations
        self.iROI = 0
        self.nROIs = 0
        self.saturation = []
        self.ROIs = []

        # saturation sliders
        self.sl = []
        txt = ["saturation", 'saturation']
        self.sat = [255,255]
        for j in range(2):
            self.sl.append(Slider(j, self))
            self.l0.addWidget(self.sl[j],1,6+5*j,1,2)
            qlabel = QtGui.QLabel(txt[j])
            qlabel.setStyleSheet('color: white;')
            self.l0.addWidget(qlabel,0,6+5*j,1,1)

        self.p1 = self.win.addPlot(name='plot1',row=1,col=0,colspan=2, title='p1')
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        self.p1.hideAxis('left')
        self.scatter1 = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter1)
        #self.p1.setLabel('bottom', 'plot1')
        #self.p1.autoRange(padding=0.01)
        self.p2 = self.win.addPlot(name='plot2',row=2,col=0,colspan=2, title='p2')
        self.p2.setMouseEnabled(x=True,y=False)
        self.p2.setMenuEnabled(False)
        self.p2.hideAxis('left')
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.addItem(self.scatter1)
        #self.p2.setLabel('bottom', 'plot2')
        self.p2.setXLink("plot1")
        #self.p2.autoRange(padding=0.01)
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No movie chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        # create ROI chooser
        #qlabel = QtGui.QLabel(self)
        #qlabel.setText("<font color='white'>Selected ROI:</font>")
        #self.l0.addWidget(qlabel,3,0,1,2)
        # create frame slider
        binLabel = QtGui.QLabel("SVD spatial bin:")
        binLabel.setStyleSheet("color: gray;")
        self.binSpinBox = QtGui.QSpinBox()
        self.binSpinBox.setRange(1, 20)
        self.binSpinBox.setValue(self.ops['sbin'])
        self.binSpinBox.setFixedWidth(50)
        self.l0.addWidget(binLabel, 7, 0, 1, 3)
        self.l0.addWidget(self.binSpinBox, 8, 0, 1, 3)
        binLabel = QtGui.QLabel("pupil sigma:")
        binLabel.setStyleSheet("color: gray;")
        self.sigmaBox = QtGui.QLineEdit()
        self.sigmaBox.setText(str(self.ops['pupil_sigma']))
        self.sigmaBox.setFixedWidth(45)
        self.l0.addWidget(binLabel, 9, 0, 1, 3)
        self.l0.addWidget(self.sigmaBox, 10, 0, 1, 3)
        self.pupil_sigma = 2.5
        self.sigmaBox.returnPressed.connect(self.pupil_sigma_change)
        self.frameLabel = QtGui.QLabel("Current frame:")
        self.frameLabel.setStyleSheet("color: white;")
        self.frameNumber = QtGui.QLabel("0")
        self.frameNumber.setStyleSheet("color: white;")
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)
        #self.frameSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.frameSlider.setTickInterval(5)
        self.frameSlider.setTracking(False)
        self.frameDelta = 10
        istretch = 23
        iplay = istretch+15
        iconSize = QtCore.QSize(20, 20)
        openButton = QtGui.QToolButton()
        openButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileIcon))
        openButton.setIconSize(iconSize)
        openButton.setToolTip("Open single movie file")
        openButton.clicked.connect(self.open_file)

        openButton2 = QtGui.QToolButton()
        openButton2.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        openButton2.setIconSize(iconSize)
        openButton2.setToolTip("Open movie folder")
        openButton2.clicked.connect(self.open_folder)

        openButton3 = QtGui.QToolButton()
        openButton3.setIcon(self.style().standardIcon(QtGui.QStyle.SP_FileDialogStart))
        openButton3.setIconSize(iconSize)
        openButton3.setToolTip("Open processed file")
        openButton3.clicked.connect(self.open_proc)

        self.process = QtGui.QPushButton('process ROIs')
        self.process.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.process.clicked.connect(self.process_ROIs)
        self.process.setEnabled(False)

        self.savefolder = QtGui.QPushButton("save folder \u2b07")
        self.savefolder.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.savefolder.clicked.connect(self.save_folder)
        self.savefolder.setEnabled(False)
        if len(self.save_path) > 0:
            self.savelabel = QtGui.QLabel(self.save_path)
        else:
            self.savelabel = QtGui.QLabel('same as video')
        self.savelabel.setStyleSheet("color: white;")

        self.saverois = QtGui.QPushButton('save ROIs')
        self.saverois.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.saverois.clicked.connect(self.save_ROIs)
        self.saverois.setEnabled(False)

        self.batchlist=[]
        self.batchname=[]
        for k in range(6):
            self.batchname.append(QtGui.QLabel(''))
            self.batchname[-1].setStyleSheet("color: white;")
            self.l0.addWidget(self.batchname[-1],18+k,0,1,4)

        self.processbatch = QtGui.QPushButton(u"process batch \u2b07")
        self.processbatch.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.processbatch.clicked.connect(self.process_batch)
        self.processbatch.setEnabled(False)

        iconSize = QtCore.QSize(30, 30)
        self.playButton = QtGui.QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPlay))
        self.playButton.setIconSize(iconSize)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QtGui.QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(iconSize)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QtGui.QButtonGroup(self)
        btns.addButton(self.playButton,0)
        btns.addButton(self.pauseButton,1)
        btns.setExclusive(True)

        quitButton = QtGui.QToolButton()
        quitButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogCloseButton))
        quitButton.setIconSize(iconSize)
        quitButton.setToolTip("Quit")
        quitButton.clicked.connect(self.close)

        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.setFixedWidth(100)
        self.comboBox.addItem("ROI type")
        self.comboBox.addItem("pupil")
        self.comboBox.addItem("motion SVD")
        self.comboBox.addItem("blink")
        self.comboBox.addItem("running")
        self.newROI = 0
        self.comboBox.setCurrentIndex(0)
        #self.comboBox.currentIndexChanged.connect(self.mode_change)

        self.addROI = QtGui.QPushButton("add ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)
        self.addROI.setEnabled(False)

        self.checkBox = QtGui.QCheckBox("Compute multivideo SVD")
        self.checkBox.setStyleSheet("color: gray;")
        if self.ops['fullSVD']:
            self.checkBox.toggle()

        self.save_mat = QtGui.QCheckBox("Save *.mat file")
        self.save_mat.setStyleSheet("color: gray;")
        if self.ops['save_mat']:
            self.save_mat.toggle()

        self.l0.addWidget(openButton,1,0,1,1)
        self.l0.addWidget(openButton2,1,1,1,1)
        self.l0.addWidget(openButton3,1,2,1,1)
        self.l0.addWidget(self.comboBox, 2, 0, 1, 3)
        self.l0.addWidget(self.addROI,3,0,1,3)
        self.l0.addWidget(self.checkBox, 11, 0, 1, 4)
        self.l0.addWidget(self.save_mat, 12, 0, 1, 3)
        self.l0.addWidget(self.savefolder, 13, 0, 1, 3)
        self.l0.addWidget(self.savelabel, 14, 0, 1, 4)
        self.l0.addWidget(self.saverois, 15, 0, 1, 3)
        self.l0.addWidget(self.process,  16, 0, 1, 3)
        self.l0.addWidget(self.processbatch,  17, 0, 1, 3)
        self.l0.addWidget(self.playButton,iplay,0,1,1)
        self.l0.addWidget(self.pauseButton,iplay,1,1,1)
        #self.l0.addWidget(quitButton,0,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)


        self.l0.addWidget(QtGui.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.frameLabel, istretch+13,0,1,3)
        self.l0.addWidget(self.frameNumber, istretch+14,0,1,3)
        self.l0.addWidget(self.frameSlider, istretch+15,3,1,15)

        # plotting boxes
        pl = QtGui.QLabel("after processing")
        pl.setStyleSheet("color: gray;")
        self.l0.addWidget(pl, istretch+1, 0, 1, 3)
        pl = QtGui.QLabel("p1")
        pl.setStyleSheet("color: gray;")
        self.l0.addWidget(pl, istretch+2, 0, 1, 1)
        pl = QtGui.QLabel("p2")
        pl.setStyleSheet("color: gray;")
        self.l0.addWidget(pl, istretch+2, 1, 1, 1)
        pl = QtGui.QLabel("roi")
        pl.setStyleSheet("color: gray;")
        self.l0.addWidget(pl, istretch+2, 2, 1, 1)
        self.cbs1 = []
        self.cbs2 = []
        self.lbls = []
        for k in range(8):
            self.cbs1.append(QtGui.QCheckBox(''))
            self.l0.addWidget(self.cbs1[-1], istretch+3+k, 0, 1, 1)
            self.cbs2.append(QtGui.QCheckBox(''))
            self.l0.addWidget(self.cbs2[-1], istretch+3+k, 1, 1, 1)
            self.cbs1[-1].toggled.connect(self.plot_processed)
            self.cbs2[-1].toggled.connect(self.plot_processed)
            self.cbs1[-1].setEnabled(False)
            self.cbs2[-1].setEnabled(False)
            self.lbls.append(QtGui.QLabel(''))
            self.lbls[-1].setStyleSheet("color: white;")
            self.l0.addWidget(self.lbls[-1], istretch+3+k, 2, 1, 1)

        #self.l0.addWidget(QtGui.QLabel(''),17,2,1,1)
        #self.l0.setRowStretch(16,2)
        ll = QtGui.QLabel('play/pause [SPACE]')
        ll.setStyleSheet("color: gray;")
        self.l0.addWidget(ll, istretch+3+k+1,0,1,4)
        #ll = QtGui.QLabel('(when paused, left/right arrow keys can move slider)')
        #ll.setStyleSheet("color: white;")
        #self.l0.addWidget(ll,14,0,1,4)
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.l0.addWidget(self.movieLabel,0,0,1,5)
        self.updateFrameSlider()
        #self.updateButtons()
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        self.cframe = 0
        self.loaded = False
        self.Floaded = False
        self.wraw = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.show()
        self.show()
        self.processed = False
        #self.load_movies([["/media/carsen/DATA2/grive/sample_movies/2016-09-29_11_M160907_MP028_eye.mj2"]])
        #self.openProc("/media/carsen/DATA1/2016-09-29_11_M160907_MP028_eye_proc.npy")
        #self.openFile(["D:/cams5/mouse_face.mp4"])
        # if not a combined recording, automatically open binary

    def pupil_sigma_change(self):
        self.pupil_sigma = float(self.sigmaBox.text())
        if len(self.ROIs) > 0:
            self.ROIs[self.iROI].plot(self)

    def add_ROI(self):
        roitype = self.comboBox.currentIndex()
        roistr = self.comboBox.currentText()
        if roitype > 0:
            self.saturation.append(255.)
            self.iROI = self.nROIs
            self.ROIs.append(roi.sROI(rind=roitype-1, rtype=roistr, iROI=self.nROIs, moveable=True, parent=self))
            self.nROIs += 1
            self.ROIs[-1].position(self)
        else:
            msg = QtGui.QMessageBox(self)
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setText("You have to choose an ROI type before creating ROI")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()

    def open_file(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                            "Open movie file")
        # load ops in same folder
        if fileName:
            print(fileName[0])
            self.filelist = [ [fileName[0]] ]
            self.load_movies()

    def save_folder(self):
        folderName = QtGui.QFileDialog.getExistingDirectory(self,
                            "Choose save folder")
        # load ops in same folder
        if folderName:
            self.save_path = folderName
            self.savelabel.setText(folderName)

    def open_folder(self):
        folderName = QtGui.QFileDialog.getExistingDirectory(self,
                            "Choose folder with movies")
        # load ops in same folder
        if folderName:
            extensions = ['*.mj2','*.mp4','*.mkv','*.avi','*.mpeg','*.mpg','*.asf']
            fileName = []
            for extension in extensions:
                fileName.extend(glob.glob(folderName+"/"+extension))
            for f in glob.glob(folderName+'/*/'):
                for extension in extensions:
                    fileName.extend(glob.glob(f+"/"+extension))
            print(fileName[0])
            if len(fileName) > 1:
                self.choose_files(fileName)
                self.load_movies()

    def choose_files(self, fileName):
        self.filelist = fileName
        LC=ListChooser('Choose movies', self)
        result = LC.exec_()
        if len(self.filelist)==0:
            self.filelist=fileName
        if len(self.filelist)>1:
            dm = QtGui.QMessageBox.question(
                self,
                "multiple videos found",
                "are you processing multiple videos taken simultaneously?",
                QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
            )
            if dm == QtGui.QMessageBox.Yes:
                print('multi camera view')
                # expects first 4 letters to be different e.g. cam0, cam1, ...
                files = []
                iview = [os.path.basename(self.filelist[0])[:4]]
                for f in self.filelist[1:]:
                    fbeg = os.path.basename(f)[:4]
                    inview = np.array([iv==fbeg for iv in iview])
                    if inview.sum()==0:
                        iview.append(fbeg)
                print(iview)
                for k in range(len(iview)):
                    ij = 0
                    for f in self.filelist:
                        if iview[k] == os.path.basename(f)[:4]:
                            if k==0:
                                files.append([])
                            files[ij].append(f)
                            ij +=1
                self.filelist = files
            else:
                print('single camera')
        else:
            self.filelist = [self.filelist]

        print(self.filelist)


    def open_proc(self):
        fileName = QtGui.QFileDialog.getOpenFileName(self,
                        "Open processed file", filter="*.npy")
        self.openProc(fileName[0])

    def openProc(self, fileName):
        try:
            proc = np.load(fileName)
            proc = proc.item()
            self.filenames = proc['filenames']
            good=True
        except:
            good=False
            print("ERROR: not a processed movie file")
        if good:
            v = []
            nframes = 0
            iframes = []
            good = self.load_movies(self.filenames)
            if good:
                if 'fullSVD' in proc:
                    self.fullSVD = proc['fullSVD']
                else:
                    self.fullSVD = True
                k=0 # number of processed things
                self.proctype = [0,0,0,0,0,0,0,0]
                self.wroi = [0,0,0,0,0,0,0,0]

                if 'motSVD' in proc:
                    self.processed = True
                else:
                    self.processed = False

                self.ROIs = []
                iROI=0
                self.typestr = ['pupil', 'motSVD', 'blink', 'run']

                if self.processed:
                    self.col = []
                    if self.fullSVD:
                        self.lbls[k].setText('fullSVD')
                        self.lbls[k].setStyleSheet("color: white;")
                        self.proctype[0] = 0
                        self.col.append((255,255,255))
                        k+=1
                    self.motSVDs = proc['motSVD']
                    self.running = proc['running']
                    self.pupil = proc['pupil']
                    self.blink = proc['blink']
                else:
                    k=1

                self.saturation = []
                kt = [0,0,0,0]
                # whether or not you can move the ROIs
                moveable = not self.processed
                if proc['rois'] is not None:
                    for r in proc['rois']:
                        rind = r['rind']
                        col = r['color']
                        yr = r['yrange']
                        xr = r['xrange']
                        ivid = r['ivid']
                        dy = r['yrange'][-1] - r['yrange'][0]
                        dx = r['xrange'][-1] - r['xrange'][0]
                        pos = [yr[0]+self.sy[ivid], xr[0]+self.sx[ivid], dy, dx]
                        self.saturation.append(r['saturation'])
                        self.ROIs.append(roi.sROI(rind=rind, rtype=r['rtype'], iROI=r['iROI'], color=r['color'],
                                         moveable=moveable, parent=self, saturation=r['saturation'],
                                         yrange=yr, xrange=xr, pos=pos, ivid=ivid))
                        self.iROI = k-1
                        self.ROIs[-1].position(self)
                        if self.processed:
                            if k < 8:
                                self.lbls[k].setText('%s%d'%(self.typestr[rind], kt[rind]))
                                r,g,b = str(int(r['color'][0])), str(int(r['color'][1])), str(int(r['color'][2]))
                                self.lbls[k].setStyleSheet("color: rgb(%s,%s,%s);"%(r,g,b))
                                self.wroi[k] = kt[rind]
                                kt[rind]+=1
                                self.proctype[k] = rind + 1
                                self.col.append(col)
                        k+=1
                self.kroi = k

                # initialize plot
                self.cframe = -1
                if self.processed:
                    for k in range(self.kroi):
                        self.cbs1[k].setEnabled(True)
                        self.cbs2[k].setEnabled(True)
                    if self.fullSVD:
                        self.cbs1[0].setChecked(True)
                    self.plot_processed()

                self.next_frame()

    def load_movies(self, filelist=None):
        if filelist is not None:
            self.filelist = filelist
        try:
            v = []
            nframes = 0
            iframes = []
            cumframes = [0]
            k=0
            for fs in self.filelist:
                vs = []
                for f in fs:
                    vs.append(pims.Video(f))
                v.append(vs)
                iframes.append(len(v[-1][0]))
                cumframes.append(cumframes[-1] + len(v[-1][0]))
                nframes += len(v[-1][0])
                if k==0:
                    Ly = []
                    Lx = []
                    for vs in v[-1]:
                        fshape = vs.frame_shape
                        Ly.append(fshape[0])
                        Lx.append(fshape[1])
                k+=1
            good = True
        except Exception as e:
            print("ERROR: not a supported movie file")
            print(e)
            good = False
        if good:
            if len(self.ROIs)>0:
                for r in self.ROIs[::-1]:
                    r.remove(self)
            self.iROI=0
            self.nROIs=0
            self.saturation=[]
            # clear checkboxes
            for k in range(len(self.lbls)):
                self.lbls[k].setText('')
                self.cbs1[k].setEnabled(False)
                self.cbs2[k].setEnabled(False)
                self.cbs1[k].setChecked(False)
                self.cbs2[k].setChecked(False)
            self.video = v
            self.filenames = self.filelist
            self.nframes = nframes
            self.iframes = np.array(iframes).astype(int)
            self.cumframes = np.array(cumframes).astype(int)
            self.Ly = Ly
            self.Lx = Lx
            self.p1.clear()
            self.p2.clear()
            if len(self.Ly)<2:
                self.LY = self.Ly[0]
                self.LX = self.Lx[0]
                self.sx = np.array([int(0)])
                self.sy = np.array([int(0)])
                self.vmap = np.zeros((self.LY,self.LX), np.int32)
            else:
                # make placement of movies
                Ly = np.array(self.Ly.copy())
                Lx = np.array(self.Lx.copy())

                LY, LX, sy, sx = video_placement(Ly, Lx)
                print(LY, LX)
                self.vmap = -1 * np.ones((LY,LX), np.int32)
                for i in range(Ly.size):
                    self.vmap[np.ix_(np.arange(sy[i], sy[i]+Ly[i], 1, int),
                                     np.arange(sx[i], sx[i]+Lx[i], 1, int))] = i
                self.sy = sy
                self.sx = sx
                self.LY = LY
                self.LX = LX

            self.fullimg = np.zeros((self.LY, self.LX, 3))
            self.imgs = []
            self.img = []
            for i in range(len(self.Ly)):
                self.imgs.append(np.zeros((self.Ly[i], self.Lx[i], 3, 3)))
                self.img.append(np.zeros((self.Ly[i], self.Lx[i], 3)))
            #self.srange = []
            # get scaling from 100 random frames in the first video
            #for n in range(len(self.Ly)):
            #    rperm = np.random.permutation(iframes[0])
            #    frames = np.zeros((self.Ly[n],self.Lx[n], min(40, iframes[0]-1)))
            #    for r in range(frames.shape[-1]):
            #        frames[:,:,r] = np.array(self.video[0][n][rperm[r]]).mean(axis=-1)
            #    self.srange.append(frames.mean() + frames.std()*np.array([-3,3]))
            self.movieLabel.setText(os.path.dirname(self.filenames[0][0]))
            self.frameDelta = int(np.maximum(5,self.nframes/200))
            self.frameSlider.setSingleStep(self.frameDelta)
            if self.nframes > 0:
                self.updateFrameSlider()
                self.updateButtons()
            self.cframe = 1
            self.loaded = True
            self.processed = False
            self.jump_to_frame()
        return good

    def keyPressEvent(self, event):
        bid = -1
        if self.playButton.isEnabled():
            if event.modifiers() !=  QtCore.Qt.ShiftModifier:
                if event.key() == QtCore.Qt.Key_Left:
                    self.cframe -= self.frameDelta
                    self.cframe  = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
                    self.frameSlider.setValue(self.cframe)
                elif event.key() == QtCore.Qt.Key_Right:
                    self.cframe += self.frameDelta
                    self.cframe  = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
                    self.frameSlider.setValue(self.cframe)
        if event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Space:
                if self.playButton.isEnabled():
                    # then play
                    self.start()
                else:
                    self.pause()

    def plot_clicked(self,event):
        items = self.win.scene().items(event.scenePos())
        posx  = 0
        posy  = 0
        iplot = 0
        zoom = False
        zoomImg = False
        choose = False
        if self.loaded:
            for x in items:
                if x==self.p1:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 1
                elif x==self.p2:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 2
                elif x==self.p0:
                    if event.button()==1:
                        if event.double():
                            zoomImg=True
                if iplot==1 or iplot==2:
                    if event.button()==1:
                        if event.double():
                            zoom=True
                        else:
                            choose=True
        if zoomImg:
            self.p0.setRange(xRange=(0,self.LX),yRange=(0,self.LY))

        if zoom:
            self.p1.setRange(xRange=(0,self.nframes))
        if choose:
            if self.playButton.isEnabled():
                self.cframe = np.maximum(0, np.minimum(self.nframes-1, int(np.round(posx))))
                self.frameSlider.setValue(self.cframe)
                #self.jump_to_frame()

    def go_to_frame(self):
        self.cframe = int(self.frameSlider.value())
        self.jump_to_frame()

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def updateFrameSlider(self):
        self.frameSlider.setMaximum(self.nframes-1)
        self.frameSlider.setMinimum(0)
        self.frameLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def updateButtons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.addROI.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.process.setEnabled(True)
        self.savefolder.setEnabled(True)
        self.saverois.setEnabled(True)

    def jump_to_frame(self):
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
            self.cframe = int(self.cframe)
            self.cframe -= 1
            self.img = self.get_frame(self.cframe)
            for i in range(len(self.img)):
                self.imgs[i][:,:,:,1] = self.img[i].copy()
            img = self.get_frame(self.cframe+1)
            for i in range(len(self.img)):
                self.imgs[i][:,:,:,2] = img[i]
            self.next_frame()

    def get_frame(self, cframe):
        cframe = np.maximum(0, np.minimum(self.nframes-1, cframe))
        cframe = int(cframe)
        try:
            ivid = (self.cumframes < cframe).nonzero()[0][-1]
        except:
            ivid = 0
        img = []
        for vs in self.video[ivid]:
            img.append(np.array(vs[cframe - self.cumframes[ivid]]))
        return img

    def next_frame(self):
        # loop after video finishes
        self.cframe+=1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
        for i in range(len(self.imgs)):
            self.imgs[i][:,:,:,:2] = self.imgs[i][:,:,:,1:]
        im = self.get_frame(self.cframe+1)
        for i in range(len(self.imgs)):
            self.imgs[i][:,:,:,2] = im[i]
            self.img[i] = self.imgs[i][:,:,:,1].copy()
            self.fullimg[np.ix_(np.arange(self.sy[i], self.sy[i]+self.Ly[i], 1, int),
                                np.arange(self.sx[i], self.sx[i]+self.Lx[i], 1, int),
                                np.arange(0, 3, 1, int))] = self.img[i]#(self.img[i].astype(np.float32) - self.srange[i][0]) / (self.img[i] - self.srange[i][1]) * 255

        if len(self.ROIs) > 0:
            self.ROIs[self.iROI].plot(self)

        self.pimg.setImage(self.fullimg)
        self.pimg.setLevels([0,self.sat[0]])
        #self.pROIimg.setLevels([0,self.sat[1]])
        self.frameSlider.setValue(self.cframe)
        self.frameNumber.setText(str(self.cframe))
        if self.processed:
            self.plot_scatter()
        self.win.show()
        self.show()

    def start(self):
        if self.cframe < self.nframes - 1:
            #print('playing')
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(25)

    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
        #print('paused')

    def save_ops(self):
        ops = {'sbin': self.sbin, 'pupil_sigma': float(self.sigmaBox.text()),
                'save_path': self.save_path, 'fullSVD': self.checkBox.isChecked(),
                'save_mat': self.save_mat.isChecked()}
        opsfile = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'ops_user.npy')
        np.save(opsfile, ops)
        return ops

    def save_ROIs(self):
        self.sbin = int(self.binSpinBox.value())
        # save running parameters as defaults
        ops = self.save_ops()

        if len(self.save_path) > 0:
            savepath = self.save_path
        else:
            savepath = None
        print(savepath)
        if len(self.ROIs)>0:
            rois = facemap.roi_to_dict(self.ROIs)
        else:
            rois = None
        proc = {'Ly':self.Ly, 'Lx':self.Lx, 'sy': self.sy, 'sx': self.sx, 'LY':self.LY, 'LX':self.LX,
                'sbin': ops['sbin'], 'fullSVD': ops['fullSVD'], 'rois': rois,
                'save_mat': ops['save_mat'], 'save_path': ops['save_path'],
                'filenames': self.filenames, 'iframes': self.iframes}
        savename = facemap.save(proc, savepath=savepath)
        self.batchlist.append(savename)
        basename,filename = os.path.split(savename)
        filename, ext = os.path.splitext(filename)
        self.batchname[len(self.batchlist)-1].setText(filename)
        self.processbatch.setEnabled(True)

    def process_batch(self):
        files = self.batchlist
        for f in files:
            proc = np.load(f).item()
            savename = facemap.run(proc['filenames'], parent=None, proc=proc, savepath=proc['save_path'])
        if len(files)==1:
            self.openProc(savename)

    def process_ROIs(self):
        self.sbin = int(self.binSpinBox.value())
        # save running parameters as defaults
        ops = self.save_ops()

        if len(self.save_path) > 0:
            savepath = self.save_path
        else:
            savepath = None
        print(savepath)
        savename = facemap.run(self.filenames, self, savepath=savepath)
        self.openProc(savename)

    def plot_processed(self):
        #self.cframe = 0
        self.p1.clear()
        self.p2.clear()
        self.traces1 = np.zeros((0,self.nframes))
        self.traces2 = np.zeros((0,self.nframes))
        self.p1.plot(3*np.ones((self.nframes,)), pen=(0,0,0))
        self.p2.plot(3*np.ones((self.nframes,)), pen=(0,0,0))
        for k in range(len(self.cbs1)):
            if self.cbs1[k].isChecked():
                tr = self.plot_trace(1, self.proctype[k], self.wroi[k], self.col[k])
                if tr.ndim<2:
                    self.traces1 = np.concatenate((self.traces1,tr[np.newaxis,:]), axis=0)
                else:
                    self.traces1 = np.concatenate((self.traces1,tr), axis=0)
        for k in range(len(self.cbs2)):
            if self.cbs2[k].isChecked():
                tr = self.plot_trace(2, self.proctype[k], self.wroi[k], self.col[k])
                if tr.ndim<2:
                    self.traces2 = np.concatenate((self.traces2,tr[np.newaxis,:]), axis=0)
                else:
                    self.traces2 = np.concatenate((self.traces2,tr), axis=0)

        self.p1.setRange(xRange=(0,self.nframes),
                         yRange=(-8, 8),
                          padding=0.0)
        self.p2.setRange(xRange=(0,self.nframes),
                         yRange=(-8, 8),
                          padding=0.0)
        self.p1.setLimits(xMin=0,xMax=self.nframes)
        self.p2.setLimits(xMin=0,xMax=self.nframes)
        self.p1.show()
        self.p2.show()
        self.plot_scatter()
        self.jump_to_frame()

    def plot_scatter(self):
        if self.traces1.shape[0] > 0:
            ntr = self.traces1.shape[0]
            self.p1.removeItem(self.scatter1)
            self.scatter1.setData(self.cframe*np.ones((ntr,)),
                                  self.traces1[:, self.cframe],
                                  size=10, brush=pg.mkBrush(255,255,255))
            self.p1.addItem(self.scatter1)

        if self.traces2.shape[0] > 0:
            ntr = self.traces2.shape[0]
            self.p2.removeItem(self.scatter2)
            self.scatter2.setData(self.cframe*np.ones((ntr,)),
                                  self.traces2[:, self.cframe],
                                  size=10, brush=pg.mkBrush(255,255,255))
            self.p2.addItem(self.scatter2)

    def plot_trace(self, wplot, proctype, wroi, color):
        if wplot==1:
            wp = self.p1
        else:
            wp = self.p2
        if proctype==0 or proctype==2:
            # motSVD
            if proctype==0:
                ir = 0
            else:
                ir = wroi+1
            cmap = cm.get_cmap("hsv")
            nc = min(4,self.motSVDs[ir].shape[1])
            cmap = (255 * cmap(np.linspace(0,0.2,nc))).astype(int)
            norm = (self.motSVDs[ir][:,0]).std()
            tr = (self.motSVDs[ir][:,:10]**2).sum(axis=1)**0.5 / norm
            for c in np.arange(0,nc,1,int)[::-1]:
                pen = pg.mkPen(tuple(cmap[c,:]), width=1)#, style=QtCore.Qt.DashLine)
                tr2 = self.motSVDs[ir][:, c] / norm
                tr2 *= np.sign(skew(tr2))
                wp.plot(tr2,  pen=pen)
            pen = pg.mkPen(color)
            wp.plot(tr, pen=pen)
        elif proctype==1:
            pup = self.pupil[wroi]
            pen = pg.mkPen(color, width=2)
            pp=wp.plot(zscore(pup['area_smooth']) - 4, pen=pen)
            pupcom = pup['com'].copy()
            pupcom -= pupcom.mean(axis=0)
            norm = pupcom.std()
            pen = pg.mkPen((155,255,155), width=1, style=QtCore.Qt.DashLine)
            py=wp.plot(pupcom[:,0] / norm + 4, pen=pen)
            pen = pg.mkPen((0,100,0), width=1, style=QtCore.Qt.DashLine)
            px=wp.plot(pupcom[:,1] / norm + 4, pen=pen)
            tr = np.concatenate((zscore(pup['area_smooth'])[np.newaxis,:] - 4,
                                 pupcom[:,0][np.newaxis,:] / norm + 4,
                                 pupcom[:,1][np.newaxis,:] / norm + 4), axis=0)
            lg=wp.addLegend(offset=(0,0))
            lg.addItem(pp,"<font color='white'><b>area</b></font>")
            lg.addItem(py,"<font color='white'><b>ypos</b></font>")
            lg.addItem(px,"<font color='white'><b>xpos</b></font>")
        elif proctype==3:
            tr = zscore(self.blink[wroi])
            pen = pg.mkPen(color, width=2)
            wp.plot(tr, pen=pen)
        elif proctype==4:
            running = self.running[wroi]
            running *= np.sign(running.mean(axis=0))
            running -= running.min()
            running /= running.max()
            running *=16
            running -=8
            wp.plot(running[:,0], pen=color)
            wp.plot(running[:,1], pen=color)
            tr = running.T
        return tr

    def button_status(self, status):
        self.playButton.setEnabled(status)
        self.pauseButton.setEnabled(status)
        self.frameSlider.setEnabled(status)
        self.process.setEnabled(status)
        self.saverois.setEnabled(status)

def run():
    # Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(
         os.path.dirname(os.path.realpath(__file__)), "mouse.png"
    )
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)
    GUI = MainW()
    #p = GUI.palette()
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# run()
