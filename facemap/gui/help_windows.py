"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
import typing

import numpy as np
from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy.QtGui import QPixmap, QGuiApplication
from qtpy.QtWidgets import (
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..version import version_str


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
        pose_tracking_group = QGroupBox()
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
        svd_group = QGroupBox()
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


class LoadNeuralDataHelp(QDialog):
    def __init__(self, window_size, parent=None):
        super(LoadNeuralDataHelp, self).__init__(parent)
        self.setWindowTitle("Help")
        width, height = int(window_size.width() * 0.28), int(
            window_size.height() * 0.25
        )
        self.resize(width, height)
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setFixedHeight(height)
        self.scrollArea.setFixedWidth(width)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidget(self.win)
        
        text = """
            <ol>
                <li>Load neural data file (*.npy) containing an array of shape neurons x time.</li>
                <li>Select whether to view neural data as heatmap or traces (for small number of neurons).</li>
                <li>(Optional) Load neural timestamps file (*.npy) containing a 1D array.</li>
                <li>(Optional) Load behavioral timestamps file (*.npy) containing a 1D array.</li>
                <li> Note: the timestamps file are used for resampling behavioral data to neural timescale.</li>
            </ol>
            """

        label = QLabel(text, self)
        label.setStyleSheet(
            "font-size: 12pt; font-family: Arial;  text-align: center; "
        )
        label.setWordWrap(True)
        layout.addWidget(label, stretch=1)

        # Add a ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()


class AboutWindow(QDialog):
    def __init__(self, parent, window_size):
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
            Framework for predicting neural activity from mouse orofacial movements tracked using a pose estimation model. Package also includes singular value decomposition (SVD) of behavioral videos. 
            </p>
            <p>
            <b>Authors:</b> Carsen Stringer & Atika Syeda
            </p>
            <p>
            <b>Contact:</b> syedaa[at]janelia.hhmi.org, stringerc[at]janelia.hhmi.org
            <p>
            <b>License:</b> GPLv3
            </p>
            <p>
            <b>Version:</b> {version}
            </p>
            <p>
            Visit our GitHub page for more information.
            </p>
        """.format(version=version_str)
        text = QTextEdit(text, self)
        text.setStyleSheet(            
                            "font-size: 12pt; color: white; background-color: #000000;"
        )
        text.setReadOnly(True) 
        text.setFixedSize(width*0.98, height)
        layout.addWidget(text, stretch=1)

        self.show()


class PoseRefinementStep2HelpWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(PoseRefinementStep2HelpWindow, self).__init__(parent)
        width, height = int(window_size.width() * 0.3), int(window_size.height() * 0.3)
        self.resize(width * 0.95, height * 0.75)
        self.setWindowTitle("Help")
        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        text = """
            <ol>
                <li>Select base model for finetuning.</li>
                <li>Set name of finetuned model.</li>
                <li>Select whether to 'Refine current video' and the number of frames to refine from current video.</li>
                <li>Set proportion of random frames to include during training. Rest of the frames are selected based on keypoint values that lie above likelihood threshold.</li>
                <li>(Optional) Select whether to use refined keypoints from previous training.</li>
                <li>Select '+' to change hyperparameter settings for training.</li>
            </ol>
            """
        label = QTextEdit(text) #QLabel(text)
        label.setReadOnly(True)  # Make the text area read-only
        label.setStyleSheet("font-size: 12pt; color: white; background-color: black;")
        label.setHtml(text)
        label.setFixedSize(width * 0.9, height * 0.6)
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
                <li><b>Move Keypoint:</b> Hold down the left mouse button and drag the mouse to reposition a keypoint.</li>
                <li><b>Restore Keypoint:</b> Right-click to add a previously deleted keypoint at the current mouse position, matching the radio button selection.</li>
                <li><b>Delete Keypoint:</b> Use Shift + D to remove the keypoint selected in the radio buttons.</li>
            </ul>
            </p>
            <h2>Guidelines for Labeling</h2>
            <p>
            Keypoints representing various facial regions are annotated according to the illustrations provided for the side view and top view. For each viewpoint, detailed instructions for labeling different regions are outlined below:
            </p>
            <h3>Eye</h3>
            <ul>
                <li>
                EYE(TOP): Mark the upper eyelid's central point, located at the eye's highest point.
                </li>
                <li>
                EYE(BOTTOM): Identify the lower eyelid's central point, situated at the eye's lowest point.
                </li>
                <li>
                EYE(FRONT): Annotate the point at the eye's front, closest to the nose.
                </li>
                <li>
                EYE(BACK): Mark the point opposite the front eye keypoint.
                </li>
            </ul>
            <h3>Nose</h3>
            <ul>
                <li>
                NOSEBRIDGE:  Place a point at the top of the nose, aligned with the "EYE(FRONT)" keypoint.
                </li>
                <li>
                NOSE(TOP):  Place a point at the top of the nose.
                </li>
                <li>
                NOSE(TIP): Annotate the point at the tip or middle of the nose.
                </li>
                <li>
                NOSE(BOTTOM): Place a point at the nose's bottom, visible from the side. In a top view, this point corresponds to the opposite side of the "Nose (Right)" keypoint, tracking lateral movements.
                </li>
                <li>
                NOSE(R): Label this point only when viewing from the top. It tracks left/right movements.
                </li>
            </ul>
             <h3>Whiskers</h3>
            For whisker labeling, identify a set of three whiskers forming a triangular pattern as shown. Look for prominent whiskers consistently recognizable across frames. Label whiskers in a clockwise order (I->II-III) when viewed from the right side, or counterclockwise order (I->II-III) when viewed from the top/left view.
            <ul> 
                <li>
                WHISKER(I): The first whisker in the third row from the top.
                </li>
                <li>
                WHISKER(II): The first whisker in the fourth row from the top.
                </li>
                <li>
                WHISKER(III): The second whisker in the fourth row from the top.
                </li>
            </ul>
            <h3>Paw</h3>
            <ul>
                <li>
                PAW: Label this point only when visible within the frame. Select any region of the paw for labeling.
                </li>
            </ul>
            <h3>Mouth</h3>
            <ul>
                <li>
                MOUTH: Mark the point at the center of the mouth opening. Label only when visible, usually in the side view.
                </li>
                <li>
                LOWERLIP: Place a point at the bottom of the lower lip near the mouth keypoint. Label only when visible, typically from the side view.
                </li>
            </ul>
            Please follow these instructions to accurately label the keypoints for each facial region in the provided illustrations.
            """
        label = QTextEdit(text)
        label.setStyleSheet(
            "font-size: 12pt; font-family: Arial; color: white; text-align: center; padding: 15;"
        )
        label.setReadOnly(True)
        label.setFixedSize(width * 0.9, height * 0.6)
        layout.addWidget(label, alignment=QtCore.Qt.AlignLeft, stretch=1)

        # Add ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        self.ok_button.setStyleSheet(
            "background: 'black'; color: 'white'; font-size: 12pt; font-family: Arial; "
        )
        layout.addWidget(self.ok_button, alignment=QtCore.Qt.AlignCenter)

        self.show()


class NeuralModelTrainingWindow(QDialog):
    def __init__(self, parent=None, window_size=None):
        super(NeuralModelTrainingWindow, self).__init__(parent)
        width, height = int(window_size.width() * 0.3), int(window_size.height() * 0.45)
        self.resize(width, height)
        self.setWindowTitle("Help - Neural Model Training")

        self.win = QWidget(self)
        layout = QVBoxLayout()
        layout.setAlignment(QtCore.Qt.AlignCenter)
        self.win.setLayout(layout)

        self.scrollArea = QScrollArea(self)
        self.scrollArea.setFixedHeight(height)
        self.scrollArea.setFixedWidth(width)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scrollArea.setWidget(self.win)

        # Add a list of hyperparameters and their descriptions with recommended values
        main_text = """
            <h2>Instructions for Training</h2>
            <p>
            <ol>
                <li>Choose the input data to be used for neural activity prediction.</li>
                <li>Select the output of the neural activity prediction model (neural principal components or neurons).</li>
                <li>Configure the hyperparameters for training the model.</li>
                <li>Indicate whether to save the output of the neural activity prediction model as yes/no.</li>
            </ol>
            </p>
            <p>
            <h4>Output of the Model</h4>
            <ul>
                <li>The results of predicting neural activity are stored in files with the extensions *.npy and/or *.mat.</li>
                <li>The output comprises a dictionary with the following entries:</li>
                <ul>
                    <li><b>predictions:</b> a two-dimensional array containing the predicted neural activity, organized as (number of features x time)</li>
                    <li><b>test_indices:</b> a list of indices indicating which data segments were used for testing, thus determining the variance explained by the model</li>
                    <li><b>variance_explained:</b> the amount of variability accounted for by the model with respect to the test data</li>
                    <li><b>plot_extent:</b> the dimensions of the plot area employed to visualize the projected neural activity, presented in the order [x1, y1, x2, y2]</li>
                </ul>
            </ul>
            </p>
            """
        main_text = QLabel(main_text, self)
        main_text.setStyleSheet(
            "font-size: 12pt; font-family: Arial;  text-align: center; "
        )
        main_text.setWordWrap(True)
        layout.addWidget(main_text, stretch=1)

        # Add ok button to close the window
        self.ok_button = QPushButton("Ok")
        self.ok_button.clicked.connect(self.close)
        self.ok_button.setStyleSheet(
            "font-size: 12pt; font-family: Arial; "
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


class ProgressBarPopup(QDialog):
    def __init__(self, gui, window_title):
        super().__init__(gui)
        self.gui = gui
        self.setWindowTitle(window_title)
        window_size = QGuiApplication.primaryScreen().availableGeometry()
        self.setFixedSize(
            int(np.floor(window_size.width() * 0.31)),
            int(np.floor(window_size.height() * 0.31 * 0.5)),
        )
        self.verticalLayout = QVBoxLayout(self)

        self.progress_bar = QProgressBar(gui)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedSize(
            int(np.floor(window_size.width() * 0.3)),
            int(np.floor(window_size.height() * 0.3 * 0.2)),
        )
        self.progress_bar.show()
        # Add the progress bar to the dialog
        self.verticalLayout.addWidget(self.progress_bar)

        # Add a cancel button to the dialog
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.close)
        self.verticalLayout.addWidget(cancel_button)

        self.show()

    def update_progress_bar(self, message, gui_obj):
        message = message.getvalue().split("\x1b[A\n\r")[0].split("\r")[-1]
        progressBar_value = [
            int(s) for s in message.split("%")[0].split() if s.isdigit()
        ]
        if len(progressBar_value) > 0:
            progress_percentage = int(progressBar_value[0])
            self.progress_bar.setValue(progress_percentage)
            self.progress_bar.setFormat(str(progress_percentage) + " %")
        gui_obj.QApplication.processEvents()
