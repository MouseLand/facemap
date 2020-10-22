import sys, os, shutil, glob, time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from scipy.stats import zscore, skew
from matplotlib import cm
from natsort import natsorted
import pathlib
import cv2

from . import process, roi, utils, io, menus, guiparts

istr = ['pupil', 'motSVD', 'blink', 'running']

class MainW(QtGui.QMainWindow):
    def __init__(self, moviefile=None, savedir=None):
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
            self.ops = np.load(opsfile, allow_pickle=True).item()
        except:
            self.ops = {'sbin': 4, 'pupil_sigma': 2., 'fullSVD': False,
                        'save_path': '', 'save_mat': False}

        self.save_path = self.ops['save_path']

        menus.mainmenu(self)
        self.online_mode=False
        #menus.onlinemenu(self)

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
        #self.showFullScreen()
        #screen = QtGui.QDesktopWidget().screenGeometry()
        #print(screen.width(), screen.height())  
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
        txt = ["Saturation:", "Saturation:"]
        self.sat = [255,255]
        for j in range(2):
            self.sl.append(guiparts.Slider(j, self))
            self.l0.addWidget(self.sl[j],1,6+5*j,1,2)
            qlabel = QtGui.QLabel(txt[j])
            qlabel.setStyleSheet('color: white;')
            self.l0.addWidget(qlabel,0,6+5*j,1,1)
            
        # Add label to indicate saturation level    
        self.saturationLevelLabel = QtGui.QLabel(str(self.sl[0].value()))
        self.saturationLevelLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.saturationLevelLabel,0,7+5*0,1,3)
        self.roiSaturationLevelLabel = QtGui.QLabel(str(self.sl[1].value()))
        self.roiSaturationLevelLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.roiSaturationLevelLabel,0,7+5*j,1,3)

        self.sl[0].valueChanged.connect(self.setSaturationLabel)        
        self.sl[1].valueChanged.connect(self.setROISaturationLabel)
        
        # Reflector
        self.reflector = QtGui.QPushButton('Add corneal reflection')
        self.l0.addWidget(self.reflector, 1, 8+5*j, 1, 2)
        self.reflector.setEnabled(False)
        self.reflector.clicked.connect(self.add_reflectROI)
        self.rROI=[]
        self.reflectors=[]

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
        self.l0.addWidget(self.movieLabel,0,0,1,5)
        self.nframes = 0
        self.cframe = 0
        
        self.make_buttons()

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
        if moviefile is not None:
            self.load_movies([[moviefile]])
        if savedir is not None:
            self.save_path = savedir
            self.savelabel.setText(savedir)

        #self.filelist = [ ['/media/carsen/DATA1/FACES/171030/test1.mp4'] ]
        #io.load_movies(self)

    def setSaturationLabel(self):
        self.saturationLevelLabel.setText(str(self.sl[0].value()))

    def setROISaturationLabel(self):
        self.roiSaturationLevelLabel.setText(str(self.sl[1].value()))

    def make_buttons(self):
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
        self.pupil_sigma = float(self.sigmaBox.text())
        self.sigmaBox.returnPressed.connect(self.pupil_sigma_change)
        self.frameLabel = QtGui.QLabel("Frame:")
        self.frameLabel.setStyleSheet("color: white;")
        self.totalFrameLabel = QtGui.QLabel("Total frames:")
        self.totalFrameLabel.setStyleSheet("color: white;")
        #self.frameNumber = QtGui.QLabel("0")
        #self.frameNumber.setStyleSheet("color: white;")
        self.setFrame = QtGui.QLineEdit()
        self.setFrame.setMaxLength(10)
        self.setFrame.setFixedWidth(50)
        self.setFrame.textChanged[str].connect(self.setFrameChanged)
        self.totalFrameNumber = QtGui.QLabel("0")             #########
        self.totalFrameNumber.setStyleSheet("color: white;")             #########
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)      
        #self.frameSlider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.frameSlider.setTickInterval(5)
        self.frameSlider.setTracking(False)
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.frameDelta = 10
        istretch = 23
        iplay = istretch+15
        iconSize = QtCore.QSize(20, 20)

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

        self.l0.addWidget(self.comboBox, 1, 0, 1, 3)
        self.l0.addWidget(self.addROI,2,0,1,3)
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
        self.l0.addWidget(self.frameLabel, istretch+12,0,1,3)
        #self.l0.addWidget(self.frameNumber, istretch+14,0,1,3)
        self.l0.addWidget(self.setFrame, istretch+12,1,2,2)    
        self.l0.addWidget(self.totalFrameLabel, istretch+14,0,1,3)
        self.l0.addWidget(self.totalFrameNumber, istretch+14,2,1,3)
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
        self.updateFrameSlider()

    def setFrameChanged(self, text):
        self.cframe = int(float(self.setFrame.text()))
        self.jump_to_frame()

    def reset(self):
        if len(self.rROI)>0:
            for r in self.rROI:
                if len(r) > 0:
                    for rr in r:
                        rr.remove(self)
        if len(self.ROIs)>0:
            for r in self.ROIs[::-1]:
                r.remove(self)
        self.ROIs = []
        self.rROI=[]
        self.reflectors=[]
        self.saturation = []
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

    def pupil_sigma_change(self):
        self.pupil_sigma = float(self.sigmaBox.text())
        if len(self.ROIs) > 0:
            self.ROIs[self.iROI].plot(self)

    def add_reflectROI(self):
        self.rROI[self.iROI].append(roi.reflectROI(iROI=self.iROI, wROI=len(self.rROI[self.iROI]), moveable=True, parent=self))

    def add_ROI(self):
        roitype = self.comboBox.currentIndex()
        roistr = self.comboBox.currentText()
        if roitype > 0:
            if self.online_mode and roitype>1:
                msg = QtGui.QMessageBox(self)
                msg.setIcon(QtGui.QMessageBox.Warning)
                msg.setText("only pupil ROI allowed during online mode")
                msg.setStandardButtons(QtGui.QMessageBox.Ok)
                msg.exec_()
                return
            self.saturation.append(255.)
            if len(self.ROIs)>0:
                if self.ROIs[self.iROI].rind==0:
                    for i in range(len(self.rROI[self.iROI])):
                        self.pROI.removeItem(self.rROI[self.iROI][i].ROI)
            self.iROI = self.nROIs
            self.ROIs.append(roi.sROI(rind=roitype-1, rtype=roistr, iROI=self.nROIs, moveable=True, parent=self))
            self.rROI.append([])
            self.reflectors.append([])
            self.nROIs += 1
            self.ROIs[-1].position(self)
        else:
            msg = QtGui.QMessageBox(self)
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setText("You have to choose an ROI type before creating ROI")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()
            return

    def save_folder(self):
        folderName = QtGui.QFileDialog.getExistingDirectory(self,
                            "Choose save folder")
        # load ops in same folder
        if folderName:
            self.save_path = folderName
            self.savelabel.setText(folderName)

    
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

    def plot_clicked(self, event):
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
            if self.playButton.isEnabled() and not self.online_mode:
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
        self.totalFrameLabel.setEnabled(True)
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
            frame_ind = cframe - self.cumframes[ivid]
            capture = vs
            if int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != frame_ind:
                capture.set(cv2.CAP_PROP_POS_FRAMES, frame_ind)
            ret, frame = capture.read()
            if ret:
                img.append(frame)
            else:
                print("Error reading frame")    
        return img

    def next_frame(self):
        if not self.online_mode:
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
                self.fullimg[self.sy[i]:self.sy[i]+self.Ly[i],
                            self.sx[i]:self.sx[i]+self.Lx[i]] = self.img[i]
            self.frameSlider.setValue(self.cframe)
            if self.processed:
                self.plot_scatter()
        else:
            self.online_plotted = False
            #online.get_frame(self)

        if len(self.ROIs) > 0:
            self.ROIs[self.iROI].plot(self)

        self.pimg.setImage(self.fullimg)
        self.pimg.setLevels([0,self.sat[0]])
        self.setFrame.setText(str(self.cframe))
        #self.frameNumber.setText(str(self.cframe))
        self.totalFrameNumber.setText(str(self.nframes))
        self.win.show()
        self.show()

    def start(self):
        if self.online_mode:
            self.online_traces = None 
            self.p1.clear()
            self.p1.show()
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(25)
        elif self.cframe < self.nframes - 1:
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
        if self.online_mode:
            self.online_traces = None
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
            rois = utils.roi_to_dict(self.ROIs, self.rROI)
        else:
            rois = None
        proc = {'Ly':self.Ly, 'Lx':self.Lx, 'sy': self.sy, 'sx': self.sx, 'LY':self.LY, 'LX':self.LX,
                'sbin': ops['sbin'], 'fullSVD': ops['fullSVD'], 'rois': rois,
                'save_mat': ops['save_mat'], 'save_path': ops['save_path'],
                'filenames': self.filenames, 'iframes': self.iframes}
        savename = process.save(proc, savepath=savepath)
        self.batchlist.append(savename)
        basename,filename = os.path.split(savename)
        filename, ext = os.path.splitext(filename)
        self.batchname[len(self.batchlist)-1].setText(filename)
        self.processbatch.setEnabled(True)

    def process_batch(self):
        files = self.batchlist
        for f in files:
            proc = np.load(f, allow_pickle=True).item()
            savename = process.run(proc['filenames'], parent=None, proc=proc, savepath=proc['save_path'])
        if len(files)==1:
            io.open_proc(self, file_name=savename)

    def process_ROIs(self):
        self.sbin = int(self.binSpinBox.value())
        # save running parameters as defaults
        ops = self.save_ops()

        if len(self.save_path) > 0:
            savepath = self.save_path
        else:
            savepath = None
        print(savepath)
        savename = process.run(self.filenames, self, savepath=savepath)
        io.open_proc(self, file_name=savename)

    def plot_processed(self):
        #self.cframe = 0
        self.p1.clear()
        self.p2.clear()
        self.traces1 = np.zeros((0,self.nframes))
        self.traces2 = np.zeros((0,self.nframes))
        #self.p1.plot(3*np.ones((self.nframes,)), pen=(0,0,0))
        #self.p2.plot(3*np.ones((self.nframes,)), pen=(0,0,0))
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
                         yRange=(-4, 4),
                          padding=0.0)
        self.p2.setRange(xRange=(0,self.nframes),
                         yRange=(-4, 4),
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
            nc = min(10,self.motSVDs[ir].shape[1])
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
            wp.setRange(yRange=(-3, 3))
        elif proctype==1:
            pup = self.pupil[wroi]
            pen = pg.mkPen(color, width=2)
            pp=wp.plot(zscore(pup['area_smooth'])*2, pen=pen)
            if 'com_smooth' in pup:
                pupcom = pup['com_smooth'].copy()
            else:
                pupcom = pup['com'].copy()
            pupcom -= pupcom.mean(axis=0)
            norm = pupcom.std()
            pen = pg.mkPen((155,255,155), width=1, style=QtCore.Qt.DashLine)
            py=wp.plot(pupcom[:,0] / norm * 2, pen=pen)
            pen = pg.mkPen((0,100,0), width=1, style=QtCore.Qt.DashLine)
            px=wp.plot(pupcom[:,1] / norm * 2, pen=pen)
            tr = np.concatenate((zscore(pup['area_smooth'])[np.newaxis,:]*2,
                                 pupcom[:,0][np.newaxis,:] / norm*2,
                                 pupcom[:,1][np.newaxis,:] / norm*2), axis=0)
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

def run(moviefile=None,savedir=None):
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
    GUI = MainW(moviefile,savedir)
    #p = GUI.palette()
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# run()
