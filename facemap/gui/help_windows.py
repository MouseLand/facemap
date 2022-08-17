import os
from ctypes import alignment

from PyQt5 import QtCore
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class MainWindowHelp(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(MainWindowHelp, self).__init__(parent)
        width, height = int(window_size.width() * 0.6), int(window_size.height() * 0.6)
        self.resize(width, height)
        self.setWindowTitle("Help")
        self.setStyleSheet("QDialog {background: 'black';}")
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        layout.setStretchFactor(self.win, 1)
        layout.setStretch(1, 1)
        self.win.setLayout(layout)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.resize(width, height)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setStyleSheet("background: 'black'; ")
        self.scrollArea.setWidget(self.win)

        # Main text section
        main_text = """
            <h1>Facemap</h1>
            <p>
            Pose tracking of mouse face from different camera views (python only) and svd processing of videos (python and MATLAB).
            </p>
            """
        main_text = QLabel(main_text, self)
        main_text.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        main_text.setWordWrap(True)
        layout.addWidget(main_text, stretch=1)

        # Pose tracking section
        pose_tracking_group = QGroupBox("Pose tracking", self)
        pose_tracking_group.setStyleSheet(
            "QGroupBox {background: 'black'; border: 0px ;}"
        )
        pose_tracking_group.setLayout(QVBoxLayout())
        pose_tracking_text = """
            <h2>Pose tracking</h2>
            <p>
            The latest python version is integrated with Facemap network for tracking 14 distinct keypoints on mouse face and an additional point for tracking paw. The keypoints can be tracked from different camera views as shown below.
            </p>
            """
        pose_tracking_text = QLabel(pose_tracking_text, self)
        pose_tracking_text.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        pose_tracking_text.setWordWrap(True)
        pose_tracking_group.layout().addWidget(pose_tracking_text, stretch=1)
        # Add images
        img_width = int(window_size.width() * 0.3)
        img_height = int(window_size.height() * 0.3)
        img_groupbox = get_img_groupbox(img_width=img_width, img_height=img_height)
        pose_tracking_group.layout().addWidget(img_groupbox, alignment=Qt.AlignCenter)

        layout.addWidget(pose_tracking_group, stretch=1)

        # SVD section
        svd_group = QGroupBox("SVD", self)
        svd_group.setStyleSheet("QGroupBox {background: 'black'; border: 0px ;}")
        svd_group.setLayout(QVBoxLayout())
        svd_text = """
            <h2>SVD processing</h2>
            <p>
            Facemap can be used to perform SVD computation on videos. This is implemented in MATLAB and Python, although the GUI is updated in Python only.
            </p>
        """
        svd_text = QLabel(svd_text)
        svd_text.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        svd_text.setWordWrap(True)
        svd_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        svd_group.layout().addWidget(svd_text, stretch=1)

        layout.addWidget(svd_group, stretch=1)

        # Add a ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        self.ok_button.setStyleSheet("background: 'black'; color: 'white'; ")
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter, stretch=1)

        self.show()

        # TODO - Add instructions for filetypes accepted for different load buttons
        # TODO - Add instructions for loading neural data


class AboutWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(AboutWindow, self).__init__(parent)
        width, height = int(window_size.width() * 0.28), int(
            window_size.height() * 0.42
        )
        self.resize(width, height)
        self.setWindowTitle("About")
        self.setStyleSheet("QDialog {background: 'black';}")
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        # Add a logo image
        logo_width = int(window_size.width() * 0.15)
        logo_height = int(window_size.height() * 0.15)
        logo_path = os.path.join(os.path.dirname(__file__), "..", "mouse.png")
        logo_pixmap = QPixmap(logo_path)
        logo_pixmap = logo_pixmap.scaled(logo_width, logo_height, Qt.KeepAspectRatio)
        logo_label = QLabel(self)
        logo_label.setPixmap(logo_pixmap)
        layout.addWidget(logo_label, alignment=QtCore.Qt.AlignCenter, stretch=1)

        # Add a text section
        header_text = """
            <h1>Facemap</h1>
            """
        header_text = QLabel(header_text, self)
        header_text.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        header_text.setWordWrap(True)
        header_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(header_text, stretch=1, alignment=QtCore.Qt.AlignCenter)

        text = """
            <p>
            Pose tracking of mouse face from different camera views (python only) and svd processing of videos (python and MATLAB).
            </p>
            <p>
            <b>Authors:</b> Carsen Stringer & Atika Syeda & Renee Tung
            </p>
            <p>
            <b>Contact:</b> syedaa[at]janelia.hhmi.org, stringerc[at]janelia.hhmi.org
            <p>
            <b>License:</b> GPLv3
            </p>
            <p>
            <b>Version:</b> 0.2.0
            </p>
            <p>
            Visit our <a href="https://github.com/MouseLand/FaceMap"> github page </a> for more information.
            </p>
        """
        text = QLabel(text, self)
        text.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; "
        )
        text.setWordWrap(True)
        text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(text, stretch=1)

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

        # Add images
        img_groupbox = get_img_groupbox(img_width=img_width, img_height=img_height)
        layout.addWidget(img_groupbox, alignment=QtCore.Qt.AlignCenter)

        text = """
            <h2>Refinement keys</h2>
            <p>
            <ul>
                <li><b>Left click:</b> To drag keypoints</li>
                <li><b>Shift + A:</b> To add keypoint selected in the radio buttons</li>
                <li><b>Shift + D:</b> To delete keypoint selected in the radio buttons</li>
            </ul>
            </p>
            <h2>Labelling instructions</h2>
            <p>
            Keypoints for different facial regions are labelled as shown above in the side view and top view. Detailed instructions for each region are given below for different views:
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
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; padding: 15;"
        )
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(label, alignment=QtCore.Qt.AlignLeft, stretch=1)

        # Add ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        self.ok_button.setStyleSheet(
            "background: 'black'; color: 'white'; font-size: 12pt; font-family: Arial; "
        )
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()


def get_img_groupbox(img_width, img_height):
    """
    Add face keypoints label examples to an image groupbox
    """

    # Create an image groupbox with horizontal layout
    img_groupbox = QGroupBox()
    img_groupbox.setLayout(QHBoxLayout())

    # Get current directory
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    # Get path that excludes last two directories
    curr_dir = os.path.join(curr_dir, "..", "..")

    # Create first image groupbox
    image1_groupbox = QGroupBox()
    image1_groupbox.setLayout(QVBoxLayout())
    image1_groupbox.layout().setSpacing(10)
    # Add header to image groupbox
    image1_header = QLabel("<h4>Side view</h4>")
    image1_header.setStyleSheet("font-size: 14pt; font-family: Arial; color: white;")
    image1_groupbox.layout().addWidget(image1_header, alignment=QtCore.Qt.AlignCenter)
    # Get path to first image
    image_path = os.path.join(curr_dir, "figs", "mouse_face1_keypoints.png")
    # resize image
    image = QLabel()
    image.setStyleSheet(
        "background: 'black'; border: 1px; border-style: solid; border-color: white;"
    )
    pixmap = QPixmap(image_path)
    pixmap = pixmap.scaled(img_width, img_height, QtCore.Qt.KeepAspectRatio)
    image.setPixmap(pixmap)
    image1_groupbox.layout().addWidget(image)
    img_groupbox.layout().addWidget(image1_groupbox, alignment=QtCore.Qt.AlignLeft)

    # Add another image to the help window
    # Create first image groupbox
    image2_groupbox = QGroupBox()
    image2_groupbox.setLayout(QVBoxLayout())
    image2_groupbox.layout().setSpacing(10)
    # Add header to image groupbox
    image2_header = QLabel("<h4>Top view</h4>")
    image2_header.setStyleSheet("font-size: 14pt; font-family: Arial; color: white;")
    image2_groupbox.layout().addWidget(image2_header, alignment=QtCore.Qt.AlignCenter)
    image_path = os.path.join(curr_dir, "figs", "mouse_face0_keypoints.png")
    # resize image
    image = QLabel()
    image.setStyleSheet(
        "background: 'black'; border: 1px; border-style: solid; border-color: white;"
    )
    pixmap = QPixmap(image_path)
    pixmap = pixmap.scaled(img_width, img_height, QtCore.Qt.KeepAspectRatio)
    image.setPixmap(pixmap)
    image2_groupbox.layout().addWidget(image)
    img_groupbox.layout().addWidget(image2_groupbox, alignment=QtCore.Qt.AlignRight)

    return img_groupbox
