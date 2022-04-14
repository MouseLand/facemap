import os
from PyQt5 import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QDialog,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QGroupBox,
    QSizePolicy,
    QScrollArea,
)


class MainWindowHelp(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(MainWindowHelp, self).__init__(parent)
        width, height = int(window_size.width() * 0.5), int(window_size.height() * 0.5)
        self.resize(width, height)
        self.setWindowTitle("Help")
        self.setStyleSheet("QDialog {background: 'black';}")
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        text = """
            <h1>Facemap</h1>
            <p>
            Pose tracking of mouse face from different camera views (python only) and svd processing of videos (python and MATLAB). Includes GUI and CLI for easy use.
            </p>
            <h2>Pose tracking</h2>
            <p>
            The latest python version is integrated with Facemap network for tracking 14 distinct keypoints on mouse face and an additional point for tracking paw. The keypoints can be tracked from different camera views.
            </p>
            <h2>SVD processing</h2>
            <p>
            Facemap can be used to perform SVD computation on videos. This is implemented in MATLAB and Python, although the GUI is updated in Python only.
            </p>
        """
        label = QLabel(text)
        label.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(label, alignment=QtCore.Qt.AlignCenter, stretch=1)

        # Add a ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter, stretch=1)

        self.show()


class PoseRefinementStep2HelpWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(PoseRefinementStep2HelpWindow, self).__init__(parent)
        width, height = int(window_size.width() * 0.3), int(window_size.height() * 0.3)
        self.resize(width, height)
        self.setWindowTitle("Help")
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        text = """
            <ol>
                <li>Select the initial/base model to use for further finetuning.</li>
                <li>Set the name of output model after refinement.</li>
                <li>(If applicable) Select data files containing refined keypoints from previous training to include during current model training.</li>
                <li>Select 'Refine current video' to refine predicted keypoints for a subset of frames from the current video after selecting the number of frames. Note: the suggested number of frames for each new animal are 20-25.</li>
                <li>Select '+' to set the hyperparameters for training.</li>
            </ol>
            """
        label = QLabel(text)
        label.setStyleSheet("font-size: 12pt; font-family: Arial; color: white;")
        label.setWordWrap(True)
        layout.addWidget(label, alignment=QtCore.Qt.AlignCenter)

        # Add a ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()


class RefinementHelpWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(RefinementHelpWindow, self).__init__(parent)
        width, height = int(window_size.width() * 0.55), int(window_size.height() * 0.5)
        img_width = int(window_size.width() * 0.3)
        img_height = int(window_size.height() * 0.3)
        self.resize(width, height)
        self.setWindowTitle("Help")

        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setFixedHeight(height)
        self.scrollArea.setFixedWidth(width)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setStyleSheet("background: 'black'; ")
        self.scrollArea.setWidget(self.win)
        # layout.addWidget(self.scrollArea)

        # Create an image groupbox with horizontal layout
        self.img_groupbox = QGroupBox()
        self.img_groupbox.setLayout(QHBoxLayout())

        # Add an image to the help window
        # Get current directory
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # Get path that excludes last two directories
        curr_dir = os.path.join(curr_dir, "..", "..")
        # Get path to image
        image_path = os.path.join(curr_dir, "figs", "mouse_face1_keypoints.png")
        # resize image
        image = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(img_width, img_height, QtCore.Qt.KeepAspectRatio)
        image.setPixmap(pixmap)
        self.img_groupbox.layout().addWidget(image, alignment=QtCore.Qt.AlignLeft)

        # Add another image to the help window
        image_path = os.path.join(curr_dir, "figs", "mouse_face0_keypoints.png")
        # resize image
        image = QLabel()
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(img_width, img_height, QtCore.Qt.KeepAspectRatio)
        image.setPixmap(pixmap)
        self.img_groupbox.layout().addWidget(image, alignment=QtCore.Qt.AlignRight)

        layout.addWidget(self.img_groupbox, alignment=QtCore.Qt.AlignCenter)

        text = """
            <h2>Labelling instructions</h2>
            <p>
            Keypoints for different facial regions are labelled as follows:
            </p>
            <h3>Eye</h3>
            <ul>
                <li>
                EYE(TOP): Upper eyelid point centered at the top of the eye.
                </li>
                <li>
                EYE(BOTTOM): Lower eyelid point centered at the bottom of the eye.
                </li>
                <li>
                EYE(FRONT): Point at the front of the eye near the nose.
                </li>
                <li>
                EYE(BACK): Point opposite the front eye keypoint.
                </li>
            </ul>
            <h3>Nose</h3>
            <ul>
                <li>
                NOSEBRIDGE: Point at the top of the nose in line with the EYE(FRONT) keypoint.
                </li>
                <li>
                NOSE(TOP): Point at the top of the nose.
                </li>
                <li>
                NOSE(TIP): Point at the tip/middle of the nose.
                </li>
                <li>
                NOSE(BOTTOM): Point at the bottom of the nose if viewed from the side. If viewed from the top, this is the point opposite the nose(Right) keypoint to track left/right movements.
                </li>
                <li>
                NOSE(R): Point at the right side of the nose which is only labeled if viewed from the top. The point tracks left/right movements.
                </li>
            </ul>
             <h3>Whiskers</h3>
            To label whiskers, find a set of 3 whiskers in the triangular configuration as shown above. The easiest way to do this is to identify most prominent whiskers that are easily identifiable across frames. Whiskers are labeled in clockwise order (C1->D1-C3) when viewed from the right side and in counterclockwise order (C1->D1-C3) when viewed from the top/left view.
            <ul>
                <li>
                WHISKER(C1): The top prominent whisker.
                </li>
                <li>
                WHISKER(C2): The bottom prominent whisker opposite to whisker(D1)
                </li>
                <li>
                WHISKER(D1): The bottom prominent whisker opposite to whisker(C2)
                </li>
            </ul>
            <h3>Paw</h3>
            <ul>
                <li>
                PAW: Point is only labelled when visible in frame. Select any region of the paw to label the paw.
                </li>
            </ul>
            <h3>Mouth</h3>
            <ul>
                <li>
                MOUTH: Point indicating the center of the mouth (opening). The point is only labeled when visible (usually from the sideview).
                </li>
                <li>
                LOWERLIP: Point at the bottom of the lower lip near the mouth keypoint. The point is only labeled when visible (usually from the sideview).
                </li>
            </ul>
            """
        label = QLabel(text)
        label.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(label, alignment=QtCore.Qt.AlignLeft, stretch=1)

        # Add ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        self.ok_button.setStyleSheet(
            "background: 'black'; color: 'white'; font-size: 12pt;     font-family: Arial; "
        )
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()
