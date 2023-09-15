from PyQt5.QtWidgets import *
from PyQt5.QtCore import QUrl, Qt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtGui import *
from PyQt5 import QtMultimediaWidgets


class VideoPlayer(QWidget):

    def __init__(self, playButton, parent=None):
        super(VideoPlayer, self).__init__(parent)

        self.scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.scene)

        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.mediaPlayer.setPlaybackRate(.5)

        self.scene = QGraphicsScene(self)
        self.graphics_view = QGraphicsView(self.scene)

        self.videoWidget = QtMultimediaWidgets.QGraphicsVideoItem() #QVideoWidget()

        self.play_button = playButton
        #self.playButton.setEnabled(False)
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_video)

        self.positionSlider = QSlider(Qt.Horizontal)
        self.positionSlider.setTickInterval(5)
        self.positionSlider.setTracking(False)
        self.positionSlider.setRange(0, 0)
        self.positionSlider.sliderMoved.connect(self.setPosition)

        self.statusBar = QStatusBar()
        self.statusBar.setFont(QFont("Arial", 10))
        # Set text color to white
        self.statusBar.setStyleSheet("color: white;")
        self.statusBar.setFixedHeight(14)

        layout = QVBoxLayout()
        self.scene.addItem(self.videoWidget)
        layout.addWidget(self.graphics_view)
        layout.addWidget(self.positionSlider)
        layout.addWidget(self.statusBar)

        self.setLayout(layout)

        self.mediaPlayer.setVideoOutput(self.videoWidget)
        self.mediaPlayer.stateChanged.connect(self.mediaStateChanged)
        self.mediaPlayer.positionChanged.connect(self.positionChanged)
        self.mediaPlayer.durationChanged.connect(self.durationChanged)
        self.mediaPlayer.error.connect(self.handleError)
        self.statusBar.showMessage("Ready")

    def abrir(self, filename):
        self.mediaPlayer.setMedia(
                QMediaContent(QUrl.fromLocalFile(filename)))
        self.play_button.setEnabled(True)
        self.statusBar.showMessage(filename)
        self.mediaPlayer.play()
        self.play_video()

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



