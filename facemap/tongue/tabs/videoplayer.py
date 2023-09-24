from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent, QVideoFrame
from PyQt5.QtGui import *
from PyQt5 import QtMultimediaWidgets, QtCore, QtMultimedia                                                
import pyqtgraph as pg
from facemap import utils

class VideoPlayer(QWidget):

    def __init__(self, playButton, parent=None):
        super(VideoPlayer, self).__init__(parent)

        #self.scene = QGraphicsScene(self)
        #self.graphics_view = QGraphicsView(self)
        #self.graphics_view.setScene(self.scene)
        self.video_window = pg.GraphicsLayoutWidget()
        #self.mediaPlayer = QMediaPlayer(self)
        #self.mediaPlayer.setPlaybackRate(.5)
        #self.videoWidget = QtMultimediaWidgets.QGraphicsVideoItem() #QVideoWidget()
                # A plot area (ViewBox + axes) for displaying the image
        self.p0 = self.video_window.addViewBox(lockAspect=True, row=0, col=0, invertY=True)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.addItem(self.pimg)
        #self.pixmap = QPixmap() #QGraphicsPixmapItem(self.videoWidget)          
        #self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.show_masks = False                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

        self.play_button = playButton
        #self.playButton.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_clicked)
        self.play_status = False

        self.current_frame = 0

        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Arial", 10))
        # Set text color to white
        self.statusBar.setStyleSheet("color: white;")
        self.statusBar.setFixedHeight(14)

        self.frameCounter = QLabel("0")
        self.frameCounter.setStyleSheet("color: white;")
        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.sliderMoved.connect(self.jump_to_frame)
        self.positionSlider.setTickInterval(5)
        self.positionSlider.setTracking(False)

        layout = QVBoxLayout()
        #self.scene.addItem(self.videoWidget)
        #self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.video_window)#graphics_view)
        layout.addWidget(self.positionSlider)
        layout.addWidget(self.frameCounter)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)

        """
        self.probe = QtMultimedia.QVideoProbe()
        self.probe.videoFrameProbed.connect(self.frameCounter.processFrame)
        self.probe.setSource(self.mediaPlayer)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        """
        self.statusBar.showMessage("Ready")

    def play_clicked(self):
        # check if current icon is paused
        if self.play_button.icon().name() == "media-playback-pause":
            self.pause()
        else:
            self.updateTimer.start(50)  
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def pause(self):
        self.updateTimer.stop()
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))

    def jump_to_frame(self):
        self.next_frame(self.positionSlider.value())

    def next_frame(self, idx=None):
        if idx is None:
            self.current_frame = self.current_frame + 1
            self.positionSlider.setValue(self.current_frame)
        else:
            self.current_frame = idx
        frame = utils.get_frame(self.current_frame, self.cumframes[-1], self.cumframes, self.containers)[0].squeeze()
        frame = frame.transpose(1, 0, 2)
        self.pimg.setImage(frame)
        self.frameCounter.setText(str(self.current_frame))


    def load_video(self, cumframes, Ly, Lx, containers):
        self.cumframes = cumframes
        self.Ly = Ly
        self.Lx = Lx
        self.containers = containers
        self.current_frame = 0
        self.play_status = True
        self.positionSlider.setRange(0, self.cumframes[-1])
        self.play_button.click()

    def update_segmentation(self):
        #self.scene.removeItem(self.pixmap_item)
        mask_image = self.array_to_qpixmap(self.masks[self.current_frame].squeeze())
        self.pixmap = QPixmap(mask_image)
        #self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.pixmap_item.setPixmap(self.pixmap)
        #self.scene.addItem(self.pixmap_item)

    def display_segmentation(self, masks, edges):
        self.show_masks = True
        self.masks = masks

    def array_to_qpixmap(self, array):
        height, width = array.shape
        q_image = QImage(width, height, QImage.Format_ARGB32)  # Use ARGB32 format for transparency

        for y in range(height):
            for x in range(width):
                value = array[y, x]
                color = QColor(255, 255, 255, 0 if value == 0 else 255)  # Set alpha to 0 for 0 values
                q_image.setPixel(x, y, color.rgba())

        return QPixmap.fromImage(q_image)

    """
    def abrir(self, filename):
        self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(filename)))
        self.play_button.setEnabled(True)
        self.statusBar.showMessage(filename)
        #self.mediaPlayer.play()
        #self.play_video()

    def play_video(self):
        self.graphics_view.fitInView(self.videoWidget, Qt.KeepAspectRatio)
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.mediaPlayer.pause()
        else:
            self.mediaPlayer.play()

    def mediaStateChanged(self, state):
        if self.mediaPlayer.state() == QMediaPlayer.PlayingState:
            self.play_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPause))
        # check if end of media state then change position to 0
        elif self.mediaPlayer.state() == QMediaPlayer.StoppedState:
            self.setPosition(0)
            self.positionChanged(0)
        else:
            self.play_button.setIcon(
                    self.style().standardIcon(QStyle.SP_MediaPlay))

    def positionChanged(self, position):
        self.positionSlider.setValue(position)

    def durationChanged(self, duration):
        self.positionSlider.setRange(0, duration)

        def setPosition(self, position):
        self.mediaPlayer.setPosition(position)

    def handleError(self):
        self.play_button.setEnabled(False)
        self.statusBar.showMessage("Error: " + self.mediaPlayer.errorString())
    """


