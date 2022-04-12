import os
from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QGroupBox,
)

class PoseRefinementStep2HelpWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(PoseRefinementStep2HelpWindow, self).__init__(parent)
        width, height = int(window_size.width()* 0.3), int(window_size.height()* 0.3)
        self.resize(width, height)
        self.setWindowTitle('Help')
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)
        
        text = ('''
            <ol>
                <li>Select the initial/base model to use for further finetuning.</li>
                <li>Set the name of output model after refinement.</li>
                <li>(If applicable) Select data files containing refined keypoints from previous training to include during current model training.</li>
                <li>Select 'Refine current video' to refine predicted keypoints for a subset of frames from the current video after selecting the number of frames. Note: the suggested number of frames for each new animal are 20-25.</li>
                <li>Select '+' to set the hyperparameters for training.</li>
            </ol>
            ''')
        label = QLabel(text)
        label.setStyleSheet("font-size: 12pt; font-family: Arial; color: white;")
        label.setWordWrap(True)
        layout.addWidget(label, alignment=QtCore.Qt.AlignCenter)

        # Add a ok button to close the window
        self.ok_button = QPushButton('Ok')
        self.ok_button.clicked.connect(self.close)
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()

class RefinementHelpWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(RefinementHelpWindow, self).__init__(parent)
        width, height = int(window_size.width()* 0.35), int(window_size.height()* 0.3)
        img_width, img_height = int(window_size.width()* 0.2), int(window_size.height()* 0.2)
        self.resize(width, height)
        self.setWindowTitle('Help')

        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)
        
        # Create an image groupbox with horizontal layout
        self.img_groupbox = QGroupBox()
        self.img_groupbox.setLayout(QHBoxLayout())

        # Add an image to the help window
        # Get current directory 
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # Get path that excludes last two directories
        curr_dir = os.path.join(curr_dir, '..', '..')
        # Get path to image
        image_path = os.path.join(curr_dir, 'figs', 'mouse_face1_keypoints.png')
        # resize image
        image = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(img_width, img_height, QtCore.Qt.KeepAspectRatio)
        image.setPixmap(pixmap)
        self.img_groupbox.layout().addWidget(image, alignment=QtCore.Qt.AlignLeft)

        # Add another image to the help window
        image_path = os.path.join(curr_dir, 'figs', 'mouse_face0_keypoints.png')    
        # resize image
        image = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(img_width, img_height, QtCore.Qt.KeepAspectRatio)
        image.setPixmap(pixmap)
        self.img_groupbox.layout().addWidget(image, alignment=QtCore.Qt.AlignRight)

        layout.addWidget(self.img_groupbox, alignment=QtCore.Qt.AlignCenter)

        # Add ok button to close the window
        self.ok_button = QPushButton('Ok')
        self.ok_button.clicked.connect(self.close)
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()

