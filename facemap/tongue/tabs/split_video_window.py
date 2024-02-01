from PyQt5.QtWidgets import *
from pyqtgraph import *
from PyQt5.QtCore import Qt

class SplitVideoWindow(QWidget):
    def __init__(self, parent, video_player):
        super().__init__()

        self.light_stylesheet = ""  # Replace with your stylesheet
        self.video_player = video_player
        self.parent = parent

        self.setWindowTitle("Split video")
        self.setWindowModality(Qt.ApplicationModal)
        self.resize(800, 600)
        self.move(QApplication.desktop().screen().rect().center() - self.rect().center())

        # Set up the layout
        split_video_window = GraphicsLayoutWidget()
        p0 = split_video_window.addViewBox(lockAspect=True, row=0, col=0, invertY=True)
        p0.setMenuEnabled(False)
        pimg = ImageItem()
        p0.addItem(pimg)
        # Set pimg to video_player's current frame
        pimg.setImage(self.video_player.pimg.image)

        self.split_video_line = InfiniteLine(angle=90, movable=True, pen=mkPen(color='w', style=Qt.DashLine, width=2))
        self.split_video_line.setPos(pimg.image.shape[0]//2)
        self.split_video_line.setBounds([0, pimg.image.shape[0]])
        self.split_video_line.sigPositionChangeFinished.connect(lambda: self.update_split_spinbox(self.split_video_line.value()))
        p0.addItem(self.split_video_line)

        # Add a text linedit and text label below the image
        splitter_groupbox = QGroupBox()
        splitter_layout = QHBoxLayout()
        splitter_groupbox.setLayout(splitter_layout)
        split_video_label = QLabel("Split frame at (x pos):")
        self.split_video_spinbox = QSpinBox()
        self.split_video_spinbox.setRange(0, pimg.image.shape[0])
        self.split_video_spinbox.setValue(pimg.image.shape[0]//2)
        self.split_video_spinbox.valueChanged.connect(lambda: self.move_splitter(self.split_video_spinbox.value()))
        splitter_layout.addWidget(split_video_label)
        splitter_layout.addWidget(self.split_video_spinbox)

        # Add a cancel and split button
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        split_button = QPushButton("Split")
        split_button.clicked.connect(self.split_video)
        button_layout = QHBoxLayout()
        button_layout.addWidget(cancel_button)
        button_layout.addWidget(split_button)

        layout = QVBoxLayout()
        layout.addWidget(split_video_window)
        layout.addWidget(splitter_groupbox)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def move_splitter(self, value):
        self.split_video_line.setPos(value)

    def update_split_spinbox(self, value):
        self.split_video_spinbox.setValue(value)

    def split_video(self):
        self.parent.add_video2(self.split_video_line.value())        
        self.close()
