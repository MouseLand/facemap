import sys
import os
import shutil
import time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
import pims
from FaceMap import facemap, roi
from scipy.stats import zscore, skew
from matplotlib import cm

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
        self.l0.addWidget(self.win,1,1,13,14)
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

        self.p1 = self.win.addPlot(name='plot1',row=1,col=0,colspan=2)
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        #self.p1.autoRange(padding=0.01)
        self.p2 = self.win.addPlot(name='plot2',row=2,col=0,colspan=2)
        self.p2.setMouseEnabled(x=True,y=False)
        self.p2.setMenuEnabled(False)
        self.p2.setXLink("plot1")
        #self.p2.autoRange(padding=0.01)
        self.win.ci.layout.setRowStretchFactor(0,5)
        self.movieLabel = QtGui.QLabel("No movie chosen")
        self.movieLabel.setStyleSheet("color: white;")
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        self.createButtons()
        # create ROI chooser
        #qlabel = QtGui.QLabel(self)
        #qlabel.setText("<font color='white'>Selected ROI:</font>")
        #self.l0.addWidget(qlabel,3,0,1,2)
        # create frame slider
        binLabel = QtGui.QLabel("SVD spatial bin:")
        binLabel.setStyleSheet("color: white;")
        self.binSpinBox = QtGui.QSpinBox()
        self.binSpinBox.setRange(1, 20)
        self.binSpinBox.setValue(4)
        self.binSpinBox.setFixedWidth(30)
        self.l0.addWidget(binLabel, 7, 0, 1, 2)
        self.l0.addWidget(self.binSpinBox, 8, 0, 1, 2)
        binLabel = QtGui.QLabel("pupil sigma:")
        binLabel.setStyleSheet("color: white;")
        self.sigmaBox = QtGui.QLineEdit()
        self.sigmaBox.setText("2.5")
        self.sigmaBox.setFixedWidth(45)
        self.l0.addWidget(binLabel, 9, 0, 1, 2)
        self.l0.addWidget(self.sigmaBox, 10, 0, 1, 2)
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
        self.l0.addWidget(QtGui.QLabel(''),12,0,1,1)
        self.l0.setRowStretch(12,1)
        self.l0.addWidget(self.frameLabel, 13,0,1,2)
        self.l0.addWidget(self.frameNumber, 14,0,1,2)
        self.l0.addWidget(self.frameSlider, 13,2,14,13)
        self.l0.addWidget(QtGui.QLabel(''),16,1,1,1)
        #self.l0.setRowStretch(16,2)
        ll = QtGui.QLabel('play/pause with SPACE')
        ll.setStyleSheet("color: white;")
        self.l0.addWidget(ll,17,0,1,3)
        ll = QtGui.QLabel('(when paused, left/right arrow keys can move slider)')
        ll.setStyleSheet("color: white;")
        self.l0.addWidget(ll,18,0,1,3)
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.l0.addWidget(self.movieLabel,0,0,1,5)
        self.updateFrameSlider()
        self.updateButtons()
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
            self.ROIs.append(roi.sROI(rind=roitype, rtype=roistr, iROI=self.nROIs, parent=self))
            self.nROIs += 1
        else:
            msg = QtGui.QMessageBox(self)
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setText("You have to choose an ROI type before creating ROI")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()

    def open(self):
        open_choice = QtGui.QMessageBox.question(
            self, "Open", "opening a movie (YES) or processed movie (NO)",
                     QtGui.QMessageBox.Yes | QtGui.QMessageBox.No
        )
        if open_choice ==  QtGui.QMessageBox.Yes:
            fileName = QtGui.QFileDialog.getOpenFileName(self,
                            "Open movie file")
            # load ops in same folder
            if fileName:
                print(fileName[0])
                self.openFile([fileName[0]])
        else:
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
            print("ERROR: not a processed movie file")
        if good:
            v = []
            nframes = 0
            iframes = []
            for file in self.filenames:
                v.append(pims.Video(file))
                iframes.append(len(v[-1]))
                nframes += len(v[-1])
            self.motSVD = proc['motSVD']
            self.motSVD *= np.sign(skew(self.motSVD, axis=0))[np.newaxis,:]
            self.motStd = self.motSVD.std(axis=0)
            self.video = v
            self.nframes = nframes
            self.iframes = np.array(iframes).astype(int)
            self.Ly = self.video[0].frame_shape[0]
            self.Lx = self.video[0].frame_shape[1]
            self.p1.clear()
            self.p2.clear()
            self.process.setEnabled(True)
            # get scaling from 100 random frames
            rperm = np.random.permutation(nframes)
            frames = np.zeros((self.Ly,self.Lx,100))
            for r in range(100):
                frames[:,:,r] = np.array(self.video[0][rperm[r]]).mean(axis=-1)
            self.srange = frames.mean() + frames.std()*np.array([-3,3])
            #self.srange = [np.percentile(frames.flatten(),8), np.percentile(frames.flatten(),99)]
            self.movieLabel.setText(self.filenames[0])
            self.nbytesread = 2 * self.Ly * self.Lx
            self.frameDelta = int(np.maximum(5,self.nframes/200))
            self.frameSlider.setSingleStep(self.frameDelta)
            if self.nframes > 0:
                self.updateFrameSlider()
                self.updateButtons()
            self.cframe = -1
            self.loaded = True
            self.processed = True
            self.plot_processed()
            self.next_frame()

    def openFile(self, fileNames):
        try:
            v = []
            nframes = 0
            iframes = []
            cumframes = [0]
            for file in fileNames:
                v.append(pims.Video(file))
                iframes.append(len(v[-1]))
                cumframes.append(cumframes[-1] + len(v[-1]))
                nframes += len(v[-1])
            good = True
        except Exception as e:
            print("ERROR: not a supported movie file")
            print(e)
            good = False
        if good:
            self.video = v
            self.filenames = fileNames
            self.nframes = nframes
            self.iframes = np.array(iframes).astype(int)
            self.cumframes = np.array(cumframes).astype(int)
            self.Ly = self.video[0].frame_shape[0]
            self.Lx = self.video[0].frame_shape[1]
            self.imgs = np.zeros((self.Ly, self.Lx, 3, 3))
            self.p1.clear()
            self.p2.clear()
            self.process.setEnabled(True)
            # get scaling from 100 random frames
            rperm = np.random.permutation(nframes)
            frames = np.zeros((self.Ly,self.Lx,100))
            for r in range(100):
                frames[:,:,r] = np.array(self.video[0][rperm[r]]).mean(axis=-1)
            self.srange = frames.mean() + frames.std()*np.array([-3,3])
            #self.srange = [np.percentile(frames.flatten(),8), np.percentile(frames.flatten(),99)]
            self.movieLabel.setText(self.filenames[0])
            self.nbytesread = 2 * self.Ly * self.Lx
            self.frameDelta = int(np.maximum(5,self.nframes/200))
            self.frameSlider.setSingleStep(self.frameDelta)
            if self.nframes > 0:
                self.updateFrameSlider()
                self.updateButtons()
            self.cframe = 1
            self.loaded = True
            self.jump_to_frame()

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
            if not self.wraw:
                self.p0.setRange(xRange=(0,self.Lx),yRange=(0,self.Ly))
            else:
                self.p0.setRange(xRange=(0,self.Lx*2+max(10,int(self.Lx*.05))),yRange=(0,self.Ly))
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

    def createButtons(self):
        iconSize = QtCore.QSize(30, 30)
        openButton = QtGui.QToolButton()
        openButton.setIcon(self.style().standardIcon(QtGui.QStyle.SP_DialogOpenButton))
        openButton.setIconSize(iconSize)
        openButton.setToolTip("Open movie file/folder")
        openButton.clicked.connect(self.open)

        self.process = QtGui.QToolButton()
        self.process.setIcon(self.style().standardIcon(QtGui.QStyle.SP_ComputerIcon))
        self.process.setIconSize(iconSize)
        self.process.setToolTip("Process ROIs")
        self.process.clicked.connect(self.process_ROIs)
        self.process.setEnabled(False)

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
        self.comboBox.setFixedWidth(80)
        self.comboBox.addItem("ROI type")
        self.comboBox.addItem("pupil")
        self.comboBox.addItem("motion SVD")
        self.comboBox.addItem("blink")
        self.newROI = 0
        self.comboBox.setCurrentIndex(0)
        #self.comboBox.currentIndexChanged.connect(self.mode_change)

        self.addROI = QtGui.QPushButton("add ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)
        self.addROI.setEnabled(False)

        self.l0.addWidget(openButton,1,0,1,1)
        self.l0.addWidget(self.process,2,0,1,1)
        self.l0.addWidget(self.comboBox, 3, 0, 1, 2)
        self.l0.addWidget(self.addROI,4,0,1,2)
        self.l0.addWidget(self.playButton,15,0,1,1)
        self.l0.addWidget(self.pauseButton,15,1,1,1)
        #self.l0.addWidget(quitButton,0,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def jump_to_frame(self):
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes-1, self.cframe))
            self.cframe = int(self.cframe)
            self.cframe -= 1
            self.img = self.get_frame(self.cframe)
            self.imgs[:,:,:,1] = self.img.copy()
            self.imgs[:,:,:,2] = self.get_frame(self.cframe+1)
            self.next_frame()

    def get_frame(self, cframe):
        cframe = np.maximum(0, np.minimum(self.nframes-1, cframe))
        cframe = int(cframe)
        try:
            ivid = (self.cumframes < cframe).nonzero()[0][-1]
        except:
            ivid = 0
        img = np.array(self.video[ivid][cframe - self.cumframes[ivid]])
        return img

    def next_frame(self):
        # loop after video finishes
        self.cframe+=1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
        self.imgs[:,:,:,:2] = self.imgs[:,:,:,1:]
        self.imgs[:,:,:,2] = self.get_frame(self.cframe+1)
        self.img = self.imgs[:,:,:,1].copy()
        #self.img = np.array(self.video[0][self.cframe])
        if len(self.ROIs) > 0:
            self.ROIs[self.iROI].plot(self)
        #if self.Floaded:
        #    self.img[self.yext,self.xext,0] = self.srange[0]
        #    self.img[self.yext,self.xext,1] = self.srange[0]
        #    self.img[self.yext,self.xext,2] = (self.srange[1]) * np.ones((self.yext.size,),np.float32)
        self.pimg.setImage(self.img)
        self.pimg.setLevels([0,self.sat[0]])
        #self.pROIimg.setLevels([0,self.sat[1]])
        self.frameSlider.setValue(self.cframe)
        self.frameNumber.setText(str(self.cframe))
        if self.processed:
            self.scatter1.setData([self.cframe, self.cframe],
                                   [self.motSVD[self.cframe, 0],
                                   self.motSVD[self.cframe, 1]],
                                   size=10,brush=pg.mkBrush(255,255,255))
            #self.scatter2.setData([self.cframe, self.cframe],
            #                      [self.motSVD[self.cframe, 0] / self.motStd[0],
            #                      self.motSVD[self.cframe, 1]] / self.motStd[1],
            #                      size=10,brush=pg.mkBrush(255,255,255))

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

    def process_ROIs(self):
        self.sbin = int(self.binSpinBox.value())
        self.motSVDs, self.pupils = facemap.run(self.filenames, self)
        self.motSVD = self.motSVDs[0]
        print(self.motSVD.shape)
        self.processed = True
        self.motSVD *= np.sign(skew(self.motSVD, axis=0))[np.newaxis,:]
        self.motStd = self.motSVD.std(axis=0)
        self.plot_processed()

    def plot_processed(self):
        self.cframe = 0
        self.p1.clear()
        self.p2.clear()
        cmap = cm.get_cmap("hsv")
        nc = min(8,self.motSVD.shape[1])
        cmap = (255 * cmap(np.linspace(0,0.8,nc))).astype(int)
        for c in range(nc):
            #self.p1.plot(self.motSVD[:, c],  pen=tuple(cmap[c,:]))
            self.p1.plot(self.motSVD[:, c] / self.motStd[c],  pen=tuple(cmap[c,:]))

        motScale = self.motSVD[:,:nc] / self.motStd[:nc][np.newaxis,:]
        self.p1.setRange(xRange=(0,self.nframes),
                         yRange=(motScale.min(), motScale.max()),
                          padding=0.0)
        self.p1.setLimits(xMin=0,xMax=self.nframes)

        self.scatter1 = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter1)
        self.scatter1.setData([self.cframe, self.cframe],
                              [self.motSVD[self.cframe, 0] / self.motStd[0],
                              self.motSVD[self.cframe, 1]] / self.motStd[1],
                              size=10,brush=pg.mkBrush(255,255,255))

        self.p2.setLimits(xMin=0,xMax=self.nframes)
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.addItem(self.scatter2)
        for p in range(len(self.pupils)):
            pup = self.pupils[p]
            self.p2.plot(zscore(pup['area']))
            self.p2.plot(zscore(pup['com'][:,0]))
            self.p2.plot(zscore(pup['com'][:,1]))
            self.p2.setRange(xRange=(0,self.nframes),
                             yRange=(-2, 4),
                             padding=0.0)
        #self.scatter2.setData([self.cframe, self.cframe],
        #                       [self.motSVD[self.cframe, 0],
        #                       self.motSVD[self.cframe, 1]],
        #                       size=10,brush=pg.mkBrush(255,255,255))

        self.jump_to_frame()

    def button_status(self, status):
        self.playButton.setEnabled(status)
        self.pauseButton.setEnabled(status)
        self.frameSlider.setEnabled(status)
        self.process.setEnabled(status)

def run():
    # Always start by initializing Qt (only once per application)
    app = QtGui.QApplication(sys.argv)
    # icon_path = os.path.join(
    #     os.path.dirname(os.path.realpath(__file__)), "logo/logo.png"
    # )
    # app_icon = QtGui.QIcon()
    # app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    # app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    # app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    # app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    # app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    # app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    # app.setWindowIcon(app_icon)
    GUI = MainW()
    #p = GUI.palette()
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)


# run()
