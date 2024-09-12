from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5 import QtCore                                  
import pyqtgraph as pg
from facemap import utils

class VideoPlayer(QWidget):

    def __init__(self, playButton, parent=None):
        super(VideoPlayer, self).__init__(parent)

        self.video_window = pg.GraphicsLayoutWidget()
        self.p0 = self.video_window.addViewBox(lockAspect=True, row=0, col=0, invertY=True)
        self.p0.setMenuEnabled(False)
        self.pimg = pg.ImageItem()
        self.p0.addItem(self.pimg)
        self.mask_image = pg.ImageItem()
        self.mask_image.setZValue(10)
        self.mask_image.setPos(0,0)
        self.p0.addItem(self.mask_image)

        self.show_masks = False           
                                                                                            
        self.play_button = playButton
        self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.play_button.clicked.connect(self.play_clicked)
        self.play_status = False

        self.current_frame = 0
        self.cumframes = []
        self.Ly = []
        self.Lx = []
        self.containers = None
        self.crop = None

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
        self.positionSlider.setTracking(True)

        layout = QVBoxLayout()
        layout.addWidget(self.video_window)#graphics_view)
        layout.addWidget(self.positionSlider)
        layout.addWidget(self.frameCounter)
        layout.addWidget(self.statusBar)
        self.setLayout(layout)

        self.statusBar.showMessage("Ready")

    def play_clicked(self):
        if self.play_status:
            self.updateTimer.stop()
            self.play_status = False
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        else:
            self.updateTimer.start()  
            self.play_status = True
            self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))

    def jump_to_frame(self):
        self.next_frame(self.positionSlider.value())

    def next_frame(self, idx=None):
        if idx is None:
            self.current_frame = self.current_frame + 1
            self.positionSlider.setValue(self.current_frame)
        else:
            self.current_frame = idx
        if self.current_frame >= self.cumframes[-1]:
            self.current_frame = 0
            self.positionSlider.setValue(self.current_frame)
        if self.crop is None:
            frame = utils.get_frame(self.current_frame, self.cumframes[-1], self.cumframes, self.containers)[0].squeeze()
        else:
            frame = utils.get_frame(self.current_frame, self.cumframes[-1], self.cumframes, self.containers, self.crop)[0].squeeze()
        frame = frame.transpose(1, 0, 2)
        self.pimg.setImage(frame)
        self.frameCounter.setText(str(self.current_frame))
        if self.show_masks:
            self.update_segmentation()

    def load_video(self, cumframes, Ly, Lx, containers, crop=None):
        self.cumframes = cumframes
        self.Ly = Ly
        self.Lx = Lx
        self.containers = containers
        self.crop = crop
        self.current_frame = 0
        self.positionSlider.setRange(0, self.cumframes[-1])
        self.play_button.click()

    def update_segmentation(self):
        mask_image = self.masks[self.current_frame].squeeze().transpose(1,0)#self.array_to_qpixmap(self.masks[self.current_frame].squeeze())
        self.mask_image.setImage(mask_image)
        self.mask_image.setOpacity(0.5)
        self.mask_image.setRect(self.pimg.boundingRect())

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