import sys, os, shutil, glob, time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from scipy.stats import zscore, skew
from matplotlib import cm
from natsort import natsorted
import pathlib
import cv2
import pandas as pd
from PyQt5.QtGui import QPixmap 
from . import process, roi, utils, io, menus, guiparts, cluster

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
        self.DLC_filepath = ""

        menus.mainmenu(self)
        self.online_mode=False
        #menus.onlinemenu(self)

        self.cwidget = QtGui.QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600,0)
        self.win.resize(1000,500)
        self.l0.addWidget(self.win,1,2,27,15)
        layout = self.win.ci.layout

        # Add logo
        #icon_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "mouse.png")
        #self.logo = QPixmap(icon_path).scaled(90, 70, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        #self.logoLabel = QtGui.QLabel(self) 
        #self.logoLabel.setPixmap(self.logo) 
        #self.logoLabel.setScaledContents(True)
        #self.l0.addWidget(self.logoLabel,0,0,3,2)

        # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.win.addViewBox(lockAspect=True,row=0,col=0,invertY=True)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.addItem(self.pimg)

        # image ROI
        self.pROI = self.win.addViewBox(lockAspect=True,row=0,col=1,invertY=True)
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
        txt = ["Saturation:", "ROI Saturation:"]
        self.sat = [255,255]
        for j in range(2):
            self.sl.append(guiparts.Slider(j, self))
            self.l0.addWidget(self.sl[j],1,3+3*j,1,2)#+5*j,1,2)
            qlabel = QtGui.QLabel(txt[j])
            qlabel.setStyleSheet('color: white;')
            self.l0.addWidget(qlabel,0,3+3*j,1,1)
        self.sl[0].valueChanged.connect(self.set_saturation_label)        
        self.sl[1].valueChanged.connect(self.set_ROI_saturation_label)

        # Add label to indicate saturation level    
        self.saturationLevelLabel = QtGui.QLabel(str(self.sl[0].value()))
        self.saturationLevelLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.saturationLevelLabel,0,5,1,1)
        self.roiSaturationLevelLabel = QtGui.QLabel(str(self.sl[1].value()))
        self.roiSaturationLevelLabel.setStyleSheet("color: white;")
        self.l0.addWidget(self.roiSaturationLevelLabel,0,8,1,1)
        
        # Reflector
        self.reflector = QtGui.QPushButton('Add corneal reflection')
        self.reflector.setEnabled(False)
        self.reflector.clicked.connect(self.add_reflectROI)
        self.rROI=[]
        self.reflectors=[]

        # Plots
        self.p1 = self.win.addPlot(name='plot1',row=1,col=0,colspan=2, title='Plot 1')
        self.p1.setMouseEnabled(x=True,y=False)
        self.p1.setMenuEnabled(False)
        self.p1.hideAxis('left')
        self.scatter1 = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter1)
        self.p2 = self.win.addPlot(name='plot2',row=2,col=0,colspan=2, title='Plot 2')
        self.p2.setMouseEnabled(x=True,y=False)
        self.p2.setMenuEnabled(False)
        self.p2.hideAxis('left')
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.addItem(self.scatter1)
        self.p2.setXLink("plot1")
        self.win.ci.layout.setRowStretchFactor(0,4)
        self.nframes = 0
        self.cframe = 0
        
        ## DLC plot
        self.DLC_scatterplot = pg.ScatterPlotItem(hover=True)
        self.DLC_scatterplot.sigClicked.connect(self.DLC_points_clicked)
        self.DLC_scatterplot.sigHovered.connect(self.DLC_points_hovered)
        self.make_buttons()
        
        self.ClusteringPlot = self.win.addPlot(row=0, col=1, lockAspect=True, enableMouse=False)
        self.ClusteringPlot.hideAxis('left')
        self.ClusteringPlot.hideAxis('bottom')
        self.clustering_scatterplot = pg.ScatterPlotItem(hover=True)
        #self.clustering_scatterplot.sigClicked.connect(lambda obj, ev: cluster.embeddedPointsClicked(obj, ev, self))
        self.clustering_scatterplot.sigHovered.connect(lambda obj, ev: self.cluster_model.embedded_points_hovered(obj, ev, parent=self))
        #self.ClusteringPlot.scene().sigMouseMoved.connect(lambda pos: self.cluster_model.mouse_moved_embedding(pos, parent=self))

        self.ClusteringPlot_legend = pg.LegendItem(labelTextSize='12pt', title="Cluster")
        self.cluster_model = cluster.Cluster(parent=self)
        self.is_cluster_labels_loaded = False
        self.loaded_cluster_labels = None

        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        self.cframe = 0
        self.loaded = False
        self.wraw = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.show()
        self.show()
        self.processed = False
        if moviefile is not None:
            self.load_movies([[moviefile]])
        if savedir is not None:
            self.save_path = savedir
            self.savelabel.setText("..."+savedir[-20:])

        # Status bar
        self.statusBar = QtGui.QStatusBar()
        self.setStatusBar(self.statusBar)
        self.progressBar = QtGui.QProgressBar()
        self.statusBar.addPermanentWidget(self.progressBar)
        self.progressBar.setGeometry(0, 0, 300, 25)
        self.progressBar.setMaximum(100)
        self.progressBar.hide()

    def set_saturation_label(self):
        self.saturationLevelLabel.setText(str(self.sl[0].value()))

    def set_ROI_saturation_label(self, val=None):
        if val is None:
            self.roiSaturationLevelLabel.setText(str(self.sl[1].value()))
        else:
            self.roiSaturationLevelLabel.setText(str(int(val)))

    def make_buttons(self):
        # create frame slider
        VideoLabel = QtGui.QLabel("Analyze Videos")
        VideoLabel.setStyleSheet("color: white;")
        VideoLabel.setAlignment(QtCore.Qt.AlignCenter)
        VideoLabel.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        #fileIOlabel = QtGui.QLabel("File I/O")
        #fileIOlabel.setStyleSheet("color: white;")
        #fileIOlabel.setAlignment(QtCore.Qt.AlignCenter)
        #fileIOlabel.setFont(QtGui.QFont("Arial", 12, QtGui.QFont.Bold))
        SVDbinLabel = QtGui.QLabel("SVD spatial bin:")
        SVDbinLabel.setStyleSheet("color: gray;")
        self.binSpinBox = QtGui.QSpinBox()
        self.binSpinBox.setRange(1, 20)
        self.binSpinBox.setValue(self.ops['sbin'])
        self.binSpinBox.setFixedWidth(50)
        binLabel = QtGui.QLabel("Pupil sigma:")
        binLabel.setStyleSheet("color: gray;")
        self.sigmaBox = QtGui.QLineEdit()
        self.sigmaBox.setText(str(self.ops['pupil_sigma']))
        self.sigmaBox.setFixedWidth(45)
        self.pupil_sigma = float(self.sigmaBox.text())
        self.sigmaBox.returnPressed.connect(self.pupil_sigma_change)
        self.frameLabel = QtGui.QLabel("Frame:")
        self.frameLabel.setStyleSheet("color: white;")
        self.totalFrameLabel = QtGui.QLabel("Total frames:")
        self.totalFrameLabel.setStyleSheet("color: white;")
        self.setFrame = QtGui.QLineEdit()
        self.setFrame.setMaxLength(10)
        self.setFrame.setFixedWidth(50)
        self.setFrame.textChanged[str].connect(self.set_frame_changed)
        self.totalFrameNumber = QtGui.QLabel("0")             #########
        self.totalFrameNumber.setStyleSheet("color: white;")             #########
        self.frameSlider = QtGui.QSlider(QtCore.Qt.Horizontal)      
        self.frameSlider.setTickInterval(5)
        self.frameSlider.setTracking(False)
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.frameDelta = 10
        istretch = 19
        iplay = istretch+10
        iconSize = QtCore.QSize(20, 20)

        self.process = QtGui.QPushButton('process ROIs')
        self.process.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.process.clicked.connect(self.process_ROIs)
        self.process.setEnabled(False)

        self.savefolder = QtGui.QPushButton("Output folder \u2b07")
        self.savefolder.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.savefolder.clicked.connect(self.save_folder)
        self.savefolder.setEnabled(False)
        self.savelabel = QtGui.QLabel('same as video')
        self.savelabel.setStyleSheet("color: white;")
        self.savelabel.setAlignment(QtCore.Qt.AlignCenter)

        self.saverois = QtGui.QPushButton('save ROIs')
        self.saverois.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.saverois.clicked.connect(self.save_ROIs)
        self.saverois.setEnabled(False)

        # DLC features
        self.loadDLC = QtGui.QPushButton("Load DLC data")
        self.loadDLC.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.loadDLC.clicked.connect(self.get_DLC_file)
        self.loadDLC.setEnabled(False)
        self.DLC_file_loaded = False
        self.DLClabels_checkBox = QtGui.QCheckBox("Labels")
        self.DLClabels_checkBox.setStyleSheet("color: gray;")
        self.DLClabels_checkBox.stateChanged.connect(self.update_DLC_points)
        self.DLClabels_checkBox.setEnabled(False)

        # Process features
        self.batchlist=[]
        """
        self.batchname=[]
        for k in range(6):
            self.batchname.append(QtGui.QLabel(''))
            self.batchname[-1].setStyleSheet("color: white;")
            self.l0.addWidget(self.batchname[-1],18+k,0,1,4)
        """

        self.processbatch = QtGui.QPushButton(u"process batch \u2b07")
        self.processbatch.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.processbatch.clicked.connect(self.process_batch)
        self.processbatch.setEnabled(False)

        # Play/pause features
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

        # Add ROI features
        self.comboBox = QtGui.QComboBox(self)
        self.comboBox.setFixedWidth(100)
        self.comboBox.addItem("Select ROI")
        self.comboBox.addItem("Pupil")
        self.comboBox.addItem("motion SVD")
        self.comboBox.addItem("Blink")
        self.comboBox.addItem("Running")
        self.newROI = 0
        self.comboBox.setCurrentIndex(0)
        #self.comboBox.currentIndexChanged.connect(self.mode_change)
        self.addROI = QtGui.QPushButton("Add ROI")
        self.addROI.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.addROI.clicked.connect(self.add_ROI)
        self.addROI.setFixedWidth(70)
        self.addROI.setEnabled(False)

        # Add clustering analysis/visualization features
        self.clusteringVisComboBox = QtGui.QComboBox(self)
        self.clusteringVisComboBox.setFixedWidth(200)
        self.clusteringVisComboBox.addItem("--Select display--")
        self.clusteringVisComboBox.addItem("ROI")
        self.clusteringVisComboBox.addItem("UMAP")
        self.clusteringVisComboBox.currentIndexChanged.connect(self.vis_combobox_selection_changed)
        self.clusteringVisComboBox.setFixedWidth(140)
        self.roiVisComboBox = QtGui.QComboBox(self)
        self.roiVisComboBox.setFixedWidth(100)
        self.roiVisComboBox.hide()
        self.roiVisComboBox.activated.connect(self.display_ROI)
        self.run_clustering_button = QtGui.QPushButton("Run")
        self.run_clustering_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.run_clustering_button.clicked.connect(lambda clicked: self.cluster_model.run(clicked, self))
        self.run_clustering_button.hide()
        self.save_clustering_button = QtGui.QPushButton("Save")
        self.save_clustering_button.setFont(QtGui.QFont("Arial", 10, QtGui.QFont.Bold))
        self.save_clustering_button.clicked.connect(lambda clicked: self.cluster_model.save_dialog(clicked, self))
        self.save_clustering_button.hide()
        self.data_clustering_combobox = QtGui.QComboBox(self)
        self.data_clustering_combobox.setFixedWidth(100)
        self.data_clustering_combobox.hide()

        # Check boxes
        self.checkBox = QtGui.QCheckBox("multivideo SVD")
        self.checkBox.setStyleSheet("color: gray;")
        if self.ops['fullSVD']:
            self.checkBox.toggle()
        self.save_mat = QtGui.QCheckBox("Save *.mat")
        self.save_mat.setStyleSheet("color: gray;")
        if self.ops['save_mat']:
            self.save_mat.toggle()
        self.motSVD_checkbox = QtGui.QCheckBox("motSVD")
        self.motSVD_checkbox.setStyleSheet("color: gray;")
        self.motSVD_checkbox.setChecked(True)
        self.movSVD_checkbox = QtGui.QCheckBox("movSVD")
        self.movSVD_checkbox.setStyleSheet("color: gray;")

        # Add features to window
        self.l0.addWidget(VideoLabel,0,0,1,2)
        self.l0.addWidget(self.comboBox,1,0,1,2)
        self.l0.addWidget(self.addROI,1,1,1,1)
        self.l0.addWidget(self.reflector, 2, 0, 1, 2)
        self.l0.addWidget(SVDbinLabel, 3, 0, 1, 2)
        self.l0.addWidget(self.binSpinBox,3, 1, 1, 2)
        self.l0.addWidget(binLabel, 4, 0, 1, 1)
        self.l0.addWidget(self.sigmaBox, 4, 1, 1, 1)
        self.l0.addWidget(self.motSVD_checkbox, 5, 0, 1, 1)
        self.l0.addWidget(self.movSVD_checkbox, 5, 1, 1, 1)
        self.l0.addWidget(self.checkBox, 6, 0, 1, 1)
        self.l0.addWidget(self.save_mat, 6, 1, 1, 1)
        self.l0.addWidget(self.saverois, 7, 0, 1, 1)
        self.l0.addWidget(self.process,  7, 1, 1, 1)
        self.l0.addWidget(self.processbatch, 8, 0, 1, 1)

        self.l0.addWidget(self.savefolder, 8, 1, 1, 1)
        self.l0.addWidget(self.savelabel, 9, 0, 1, 2)
        self.l0.addWidget(self.loadDLC, 10, 0, 1, 1)                    # DLC features
        self.l0.addWidget(self.DLClabels_checkBox, 10, 1, 1, 1)        
        self.l0.addWidget(self.clusteringVisComboBox, 0, 11, 1, 1)      # clustering visualization window features
        self.l0.addWidget(self.data_clustering_combobox, 0, 12, 1, 2)      # clustering visualization window features
        self.l0.addWidget(self.roiVisComboBox, 0, 12, 1, 2)              # ROI visualization window features
        self.l0.addWidget(self.run_clustering_button, 0, 14, 1, 1)      # clustering visualization window features
        self.l0.addWidget(self.save_clustering_button, 0, 15, 1, 1)      # clustering visualization window features
        self.l0.addWidget(self.playButton,iplay,0,1,1)
        self.l0.addWidget(self.pauseButton,iplay,1,1,1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)
        self.l0.addWidget(QtGui.QLabel(''),istretch,0,1,3)
        self.l0.setRowStretch(istretch,1)
        self.l0.addWidget(self.frameLabel, istretch+7,0,1,1)
        self.l0.addWidget(self.setFrame, istretch+7,1,1,1)    
        self.l0.addWidget(self.totalFrameLabel, istretch+8,0,1,1)
        self.l0.addWidget(self.totalFrameNumber, istretch+8,1,1,1)
        self.l0.addWidget(self.frameSlider, istretch+10,2,1,15)

        # plotting boxes
        #pl = QtGui.QLabel("Plot output")
        #pl.setStyleSheet("color: white")
        #pl.setAlignment(QtCore.Qt.AlignCenter)
        #self.l0.addWidget(pl, istretch+1, 0, 1, 2)
        pl = QtGui.QLabel("Plot 1")
        pl.setStyleSheet("color: gray;")
        self.l0.addWidget(pl, istretch, 0, 1, 1)
        pl = QtGui.QLabel("Plot 2")
        pl.setStyleSheet("color: gray;")
        self.l0.addWidget(pl, istretch, 1, 1, 1)
        pl = QtGui.QLabel("ROI")
        pl.setStyleSheet("color: gray;")
        #self.l0.addWidget(pl, istretch+2, 2, 1, 1)
        self.cbs1 = []
        self.cbs2 = []
        self.lbls = []
        for k in range(5):
            self.cbs1.append(QtGui.QCheckBox(""))
            self.l0.addWidget(self.cbs1[-1], istretch+1+k, 0, 1, 1)
            self.cbs2.append(QtGui.QCheckBox(""))
            self.l0.addWidget(self.cbs2[-1], istretch+1+k, 1, 1, 1)
            self.cbs1[-1].toggled.connect(self.plot_processed)
            self.cbs2[-1].toggled.connect(self.plot_processed)
            self.cbs1[-1].setEnabled(False)
            self.cbs2[-1].setEnabled(False)
            self.cbs1[k].setStyleSheet("color: gray;")
            self.cbs2[k].setStyleSheet("color: gray;")
            self.lbls.append(QtGui.QLabel(''))
            self.lbls[-1].setStyleSheet("color: white;")
            #self.l0.addWidget(self.lbls[-1], istretch+3+k, 2, 1, 1)
        #ll = QtGui.QLabel('play/pause [SPACE]')
        #ll.setStyleSheet("color: gray;")
        #self.l0.addWidget(ll, istretch+3+k+1,0,1,1)
        self.update_frame_slider()

    def vis_combobox_selection_changed(self):
        """
        Call clustering or ROI display functions upon user selection from combo box
        """
        self.clear_visualization_window()
        visualization_request = self.clusteringVisComboBox.currentText()
        if visualization_request == "ROI":
            self.cluster_model.disable_data_clustering_features(self)
            if len(self.ROIs)>0:
                self.update_ROI_vis_comboBox()
                self.update_status_bar("")
            else:
                self.update_status_bar("Please add ROIs for display")
        elif visualization_request == "UMAP":
            self.cluster_model.enable_data_clustering_features(parent=self)
            self.update_status_bar("")
        else:
            self.cluster_model.disable_data_clustering_features(self)

    def clear_visualization_window(self):
        self.roiVisComboBox.hide()
        self.pROIimg.clear()
        self.pROI.removeItem(self.scatter)
        self.ClusteringPlot.clear()
        self.ClusteringPlot.hideAxis('left')
        self.ClusteringPlot.hideAxis('bottom')
        self.ClusteringPlot.removeItem(self.clustering_scatterplot)
        self.ClusteringPlot_legend.setParentItem(None)
        self.ClusteringPlot_legend.hide()

    def update_ROI_vis_comboBox(self):
        """
        Update ROI selection combo box
        """
        self.roiVisComboBox.clear()
        self.pROIimg.clear() 
        self.roiVisComboBox.addItem("--Type--")
        for i in range(len(self.ROIs)):
            selected = self.ROIs[i]
            self.roiVisComboBox.addItem(str(selected.iROI+1)+". "+selected.rtype)
        if self.clusteringVisComboBox.currentText() == "ROI":
            self.roiVisComboBox.show()

    def display_ROI(self):
        """
        Plot selected ROI on visualizaiton window 
        """
        self.roiVisComboBox.show()
        roi_request = self.roiVisComboBox.currentText()
        if roi_request != "--Type--":
            self.pROI.addItem(self.scatter)
            roi_request_ind = int(roi_request.split(".")[0]) - 1
            self.ROIs[int(roi_request_ind)].plot(self)
            #self.set_ROI_saturation_label(self.ROIs[int(roi_request_ind)].saturation)
        else:
            self.pROIimg.clear()
            self.pROI.removeItem(self.scatter)

    def set_frame_changed(self, text):
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
        self.clear_visualization_window()
        # Clear clusters
        self.cluster_model.disable_data_clustering_features(self)
        self.clusteringVisComboBox.setCurrentIndex(0)
        self.ClusteringPlot.clear()
        # Clear DLC variables when a new file is loaded
        #self.DLCplot.clear()
        self.DLC_scatterplot.clear()
        #self.p0.clear()
        self.DLC_file_loaded = False
        # clear checkboxes
        for k in range(len(self.cbs1)):
            self.cbs1[k].setText("")
            self.cbs2[k].setText("")
            self.lbls[k].setText("")
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
            self.update_ROI_vis_comboBox()
            self.ROIs[-1].position(self)
        else:
            msg = QtGui.QMessageBox(self)
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setText("You have to choose an ROI type before creating ROI")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()
            return

    def update_status_bar(self, message, update_progress=False):
        if update_progress:
            self.progressBar.show()
            progressBar_value = [int(s) for s in message.split("%")[0].split() if s.isdigit()]
            self.progressBar.setValue(progressBar_value[0])
            frames_processed = np.floor((progressBar_value[0]/100)*float(self.totalFrameNumber.text()))
            self.setFrame.setText(str(frames_processed))
            self.statusBar.showMessage(message.split("|")[0])
        else: 
            self.progressBar.hide()
            self.statusBar.showMessage(message)

    def save_folder(self):
        folderName = QtGui.QFileDialog.getExistingDirectory(self,
                            "Choose save folder")
        # load ops in same folder
        if folderName:
            self.save_path = folderName
            if len(folderName) > 30:
                self.savelabel.setText("..."+folderName[-30:])
            else:
                self.savelabel.setText(folderName)

    def get_DLC_file(self):
        filepath = QtGui.QFileDialog.getOpenFileName(self,
                                "Choose DLC file", "", "DLC labels file (*.h5)")
        if filepath[0]:
            self.DLC_filepath = filepath[0]
            self.DLC_file_loaded = True
            self.update_status_bar("DLC file loaded: "+self.DLC_filepath)
            self.load_DLC_points()

    def load_DLC_points(self):
        # Read DLC file
        self.DLC_data = pd.read_hdf(self.DLC_filepath, 'df_with_missing')
        all_labels = self.DLC_data.columns.get_level_values("bodyparts")
        self.DLC_keypoints_labels = [all_labels[i] for i in sorted(np.unique(all_labels, return_index=True)[1])]#np.unique(self.DLC_data.columns.get_level_values("bodyparts"))
        self.DLC_x_coord = self.DLC_data.T[self.DLC_data.columns.get_level_values("coords").values=="x"].values #size: key points x frames
        self.DLC_y_coord = self.DLC_data.T[self.DLC_data.columns.get_level_values("coords").values=="y"].values #size: key points x frames
        self.DLC_likelihood = self.DLC_data.T[self.DLC_data.columns.get_level_values("coords").values=="likelihood"].values #size: key points x frames
        # Choose colors for each label: provide option for color blindness as well
        self.colors = cm.get_cmap('gist_rainbow')(np.linspace(0, 1., len(self.DLC_keypoints_labels)))
        self.colors *= 255
        self.colors = self.colors.astype(int)
        self.colors[:,-1] = 127
        self.brushes = np.array([pg.mkBrush(color=c) for c in self.colors])
    
    def update_DLC_points(self):
        if self.DLC_file_loaded and self.DLClabels_checkBox.isChecked():
            self.statusBar.clearMessage()
            self.p0.addItem(self.DLC_scatterplot)
            self.p0.setRange(xRange=(0,self.LX), yRange=(0,self.LY), padding=0.0)
            filtered_keypoints = np.where(self.DLC_likelihood[:,self.cframe] > 0.9)[0]
            x = self.DLC_x_coord[filtered_keypoints,self.cframe]
            y = self.DLC_y_coord[filtered_keypoints,self.cframe]
            self.DLC_scatterplot.setData(x, y, size=15, symbol='o', brush=self.brushes[filtered_keypoints], hoverable=True, hoverSize=15)
        elif not self.DLC_file_loaded and self.DLClabels_checkBox.isChecked():
            self.update_status_bar("Please upload a DLC (*.h5) file")
        else:
            self.statusBar.clearMessage()
            self.DLC_scatterplot.clear()

    def DLC_points_clicked(self, obj, points):
        ## Can add functionality for clicking key points
        return ""

    def DLC_points_hovered(self, obj, ev):
        point_hovered = np.where(self.DLC_scatterplot.data['hovered'])[0]
        if point_hovered.shape[0] >= 1:         # Show tooltip only when hovering over a point i.e. no empty array
            points = self.DLC_scatterplot.points()
            vb = self.DLC_scatterplot.getViewBox()
            if vb is not None and self.DLC_scatterplot.opts['tip'] is not None:
                cutoff = 1                      # Display info of only one point when hovering over multiple points
                tip = [self.DLC_scatterplot.opts['tip'](data = self.DLC_keypoints_labels[pt],x=points[pt].pos().x(), y=points[pt].pos().y())
                        for pt in point_hovered[:cutoff]]
                if len(point_hovered) > cutoff:
                    tip.append('({} other...)'.format(len(point_hovered) - cutoff))
                vb.setToolTip('\n\n'.join(tip))

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
        self.setFrame.setText(str(self.cframe))
        #self.jump_to_frame()

    def fitToWindow(self):
        self.movieLabel.setScaledContents(self.fitCheckBox.isChecked())

    def update_frame_slider(self):
        self.frameSlider.setMaximum(self.nframes-1)
        self.frameSlider.setMinimum(0)
        self.frameLabel.setEnabled(True)
        self.totalFrameLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def update_buttons(self):
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.addROI.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.process.setEnabled(True)
        self.savefolder.setEnabled(True)
        self.saverois.setEnabled(True)
        self.checkBox.setChecked(True)
        self.save_mat.setChecked(True)

        # Enable DLC features for single video only
        if len(self.img)==1:
            self.loadDLC.setEnabled(True)
            self.DLClabels_checkBox.setEnabled(True)
        else:
            self.loadDLC.setEnabled(False)
            self.DLClabels_checkBox.setEnabled(False)

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

        if len(self.ROIs) > 0 and self.clusteringVisComboBox.currentText() == "ROI":
            self.ROIs[self.iROI].plot(self)

        self.pimg.setImage(self.fullimg)
        self.pimg.setLevels([0,self.sat[0]])
        self.setFrame.setText(str(self.cframe))
        self.update_DLC_points()
        #self.frameNumber.setText(str(self.cframe))
        self.totalFrameNumber.setText(str(self.nframes))
        self.win.show()
        self.show()

    def start(self):
        if self.online_mode:
            self.online_traces = None 
            #self.p1.clear()
            self.p1.show()
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(25)
        elif self.cframe < self.nframes - 1:
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(25)
        self.update_DLC_points()

    def pause(self):
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
        if self.online_mode:
            self.online_traces = None
        self.update_DLC_points()

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
        print("ROIs saved in:", savepath)
        if len(self.ROIs)>0:
            rois = utils.roi_to_dict(self.ROIs, self.rROI)
        else:
            rois = None
        proc = {'Ly':self.Ly, 'Lx':self.Lx, 'sy': self.sy, 'sx': self.sx, 'LY':self.LY, 'LX':self.LX,
                'sbin': ops['sbin'], 'fullSVD': ops['fullSVD'], 'rois': rois,
                'save_mat': ops['save_mat'], 'save_path': ops['save_path'],
                'filenames': self.filenames}
        savename = process.save(proc, savepath=savepath)
        self.update_status_bar("File saved in "+savepath) #### 
        self.batchlist.append(savename)
        basename,filename = os.path.split(savename)
        filename, ext = os.path.splitext(filename)
        #self.batchname[len(self.batchlist)-1].setText(filename)
        self.processbatch.setEnabled(True)

    def process_batch(self):
        if self.motSVD_checkbox.isChecked() or self.movSVD_checkbox.isChecked():
            files = self.batchlist
            for f in files:
                proc = np.load(f, allow_pickle=True).item()
                savename = process.run(proc['filenames'], GUIobject=QtGui, parent=self, proc=proc, savepath=proc['save_path'])
            if len(files)==1:
                io.open_proc(self, file_name=savename)
        else:
            msg = QtGui.QMessageBox(self)
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setText("Please check at least one of: motSVD, movSVD")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()
            return

    def process_ROIs(self):
        self.sbin = int(self.binSpinBox.value())
        # save running parameters as defaults
        ops = self.save_ops()
        if len(self.save_path) > 0:
            savepath = self.save_path
        else:
            savepath = None
        if self.motSVD_checkbox.isChecked() or self.movSVD_checkbox.isChecked():
            savename = process.run(self.filenames, GUIobject=QtGui, parent=self, savepath=savepath)
            io.open_proc(self, file_name=savename)
            print("Output saved in",savepath)
            self.update_status_bar("Output saved in "+savepath)
        else: 
            msg = QtGui.QMessageBox(self)
            msg.setIcon(QtGui.QMessageBox.Warning)
            msg.setText("Please check at least one of: motSVD, movSVD")
            msg.setStandardButtons(QtGui.QMessageBox.Ok)
            msg.exec_()
            return

    def plot_processed(self):
        self.p1.clear()
        self.p2.clear()
        self.traces1 = np.zeros((0,self.nframes))
        self.traces2 = np.zeros((0,self.nframes))
        for k in range(len(self.cbs1)):
            if self.cbs1[k].isChecked():
                self.cbs1[k].setText(self.lbls[k].text())
                self.cbs1[k].setStyleSheet(self.lbls[k].styleSheet())
                tr = self.plot_trace(1, self.proctype[k], self.wroi[k], self.col[k])
                if tr.ndim<2:
                    self.traces1 = np.concatenate((self.traces1,tr[np.newaxis,:]), axis=0)
                else:
                    self.traces1 = np.concatenate((self.traces1,tr), axis=0)
            else:
                self.cbs1[k].setText(self.lbls[k].text())
                self.cbs1[k].setStyleSheet("color: gray")
        for k in range(len(self.cbs2)):
            if self.cbs2[k].isChecked():
                self.cbs2[k].setText(self.lbls[k].text())
                #print(self.lbls[k].palette().color(QtGui.QPalette.Background))
                self.cbs2[k].setStyleSheet(self.lbls[k].styleSheet())
                tr = self.plot_trace(2, self.proctype[k], self.wroi[k], self.col[k])
                if tr.ndim<2:
                    self.traces2 = np.concatenate((self.traces2,tr[np.newaxis,:]), axis=0)
                else:
                    self.traces2 = np.concatenate((self.traces2,tr), axis=0)
            else:
                self.cbs2[k].setText(self.lbls[k].text())
                self.cbs2[k].setStyleSheet("color: gray")
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
