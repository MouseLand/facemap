"""
Keypoints correction feature for new mouse videos
"""
import os
from natsort import natsorted
import pyqtgraph as pg
import numpy as np
from .. import utils
from ..gui import io
from PyQt5 import QtCore
import pandas as pd
from matplotlib import cm
from glob import glob
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QWidget, QLineEdit, QLabel, QDialogButtonBox,
                            QSpinBox, QPushButton, QVBoxLayout, QComboBox, QMessageBox,
                            QHBoxLayout, QVBoxLayout, QButtonGroup, QGroupBox,
                            QListWidget, QCheckBox, QDesktopWidget)

"""
Single workflow for re-training the model or fine-tuning the model with new data.
Trained model overwrites any previously trained model.
"""
class ModelTrainingPopup(QDialog):
    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.output_folder_path = None
        self.model_files = None
        self.data_files = None
        self.selected_videos = None
        self.use_current_video = False

        self.setWindowTitle('Model training')
        self.setStyleSheet("QDialog {background: 'black';}")

        self.verticalLayout = QVBoxLayout(self)
        # Set window size that is adjusted to the size of the window
        self.window_max_size = QDesktopWidget().screenGeometry(-1)

        self.show_choose_folder()
        self.setLayout(self.verticalLayout)

        self.show()

    def update_window_size(self, frac=0.5):
        # Set the size of the window to be a fraction of the screen size
        self.resize(int(np.floor(self.window_max_size.width()*frac)), int(np.floor(self.window_max_size.height()*frac)))

    def clear_window(self):
        # Clear the window
        for i in reversed(range(self.verticalLayout.count())): 
            self.verticalLayout.itemAt(i).widget().setParent(None)

    def update_window_title(self, title=None):
        if title is None:
            self.setWindowTitle('Keypoints refinement: frame {}/{}'.format(self.current_frame+1, self.spinBox_nframes.value()))
        else:
            self.setWindowTitle(title)

    def show_choose_folder(self):
        """
        Ask user to set path to output folder where refined keypoints and trained model will be saved
        """
        self.update_window_size(0.5)

        # Add a QLineEdit widget to the layout and a browse button to set the path to the output folder
        self.output_folder_path = QLabel(self)
        self.output_folder_path.setText("Step 1: Choose output folder path")
        self.output_folder_path.setStyleSheet("QLabel {color: 'white';}")
        self.verticalLayout.addWidget(self.output_folder_path)

        # Create a QGroupbox widget to hold the browse button and the QLineEdit widget with a horizontal layout
        self.output_folder_groupbox = QGroupBox(self)
        self.output_folder_groupbox.setLayout(QHBoxLayout())
        self.output_folder_groupbox.setStyleSheet("QGroupBox {color: 'white'; border: 0px}")

        self.output_folder_path_box = QLineEdit(self)
        self.output_folder_path_box.setStyleSheet("QLineEdit {color: 'black';}")
        self.output_folder_path_box.setText("...")
        self.output_folder_groupbox.layout().addWidget(self.output_folder_path_box)

        self.output_folder_path_button = QPushButton('Browse', self)
        self.output_folder_path_button.clicked.connect(lambda clicked: self.set_output_folder_path())
        self.output_folder_groupbox.layout().addWidget(self.output_folder_path_button)

        self.verticalLayout.addWidget(self.output_folder_groupbox)

        # Add a next button to the layout and connect it to the next step
        # Create a QGroupbox widget to hold the next button
        self.next_button_groupbox = QGroupBox(self)
        self.next_button_groupbox.setLayout(QHBoxLayout())
        self.next_button_groupbox.setStyleSheet("QGroupBox {color: 'white'; border: 0px;}")
        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(lambda clicked: self.show_choose_training_files())
        # Align the button to the right of the layout
        self.next_button_groupbox.layout().addWidget(self.next_button, alignment=QtCore.Qt.AlignRight)
        self.verticalLayout.addWidget(self.next_button_groupbox)


    def set_output_folder_path(self):
        path = io.get_folder_path(parent=self.gui)
        self.output_folder_path_box.setText(path)
        self.output_folder_path = path

    def show_choose_training_files(self):
        self.clear_window()
        self.update_window_title("Step 2: Choose training files")
        print("Path set to {}".format(self.output_folder_path))

        # Check if the selected output folder path contains a model file (*.pt) and data files (*.npy)
        self.model_files = glob(os.path.join(self.output_folder_path, '*.pt'))
        self.data_files = glob(os.path.join(self.output_folder_path, '*Facemap_refined_images_landmarks.npy'))
        if len(self.model_files) == 0 and len(self.data_files) == 0:
            print("No model or data files found in the selected output folder")
            return
        else:
            print("Model files found: {}".format(self.model_files))
            print("Data files found: {}".format(self.data_files))

        # Add a QGroupbox widget to hold a qlabel and dropdown menu
        self.model_groupbox = QGroupBox(self)
        self.model_groupbox.setLayout(QHBoxLayout())
        self.model_groupbox.setStyleSheet("QGroupBox {color: 'white'; border: 0px")

        self.model_label = QLabel(self)
        self.model_label.setText("Model:")
        self.model_label.setStyleSheet("QLabel {color: 'white';}")
        self.model_groupbox.layout().addWidget(self.model_label)

        self.model_dropdown = QComboBox(self)
        # Add the model files to the dropdown menu
        for model_file in self.model_files:
            self.model_dropdown.addItem(os.path.basename(model_file))
        self.model_dropdown.setStyleSheet("QComboBox {color: 'black';}")
        self.model_groupbox.layout().addWidget(self.model_dropdown)

        self.verticalLayout.addWidget(self.model_groupbox)

        # Add a QGroupbox widget to hold checkboxes for selecting videos using the list of data files
        self.npy_files_groupbox = QGroupBox(self)
        self.npy_files_groupbox.setLayout(QVBoxLayout())
        
        self.npy_files_label = QLabel(self)
        self.npy_files_label.setText("Data files:")
        self.npy_files_label.setStyleSheet("QLabel {color: 'white';}")
        self.npy_files_groupbox.layout().addWidget(self.npy_files_label)

        self.npy_files_checkboxes = []
        for i, file in enumerate(self.data_files):
            checkbox = QCheckBox(file, self)
            checkbox.setStyleSheet("QCheckBox {color: 'white';}")
            self.npy_files_groupbox.layout().addWidget(checkbox)
            self.npy_files_checkboxes.append(checkbox)
        
        self.verticalLayout.addWidget(self.npy_files_groupbox)

        # Add a QCheckBox widget for user to select whether to use the current video
        self.use_current_video_checkbox = QCheckBox(self)
        self.use_current_video_checkbox.setText("Use current video")
        self.use_current_video_checkbox.setStyleSheet("QCheckBox {color: 'white'; }")
        self.verticalLayout.addWidget(self.use_current_video_checkbox)

        # Add a QGroupbox widget to hold two buttons for cancel and next step with a horizontal layout aligned to the right
        self.buttons_groupbox = QGroupBox(self)
        self.buttons_groupbox.setLayout(QHBoxLayout())
        self.buttons_groupbox.setStyleSheet("QGroupBox {color: 'white'; border: 0px}")

        self.cancel_button = QPushButton('Cancel', self)
        self.cancel_button.clicked.connect(lambda clicked: self.close())
        self.buttons_groupbox.layout().addWidget(self.cancel_button, alignment=QtCore.Qt.AlignRight)

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(lambda clicked: self.update_user_training_options())
        self.buttons_groupbox.layout().addWidget(self.next_button, alignment=QtCore.Qt.AlignRight)

        self.verticalLayout.addWidget(self.buttons_groupbox)

    def update_user_training_options(self):
        # Get the selected model
        self.model_files = self.model_files[self.model_dropdown.currentIndex()]
        # Get the selected videos
        self.selected_videos = []
        for i, checkbox in enumerate(self.npy_files_checkboxes):
            if checkbox.isChecked():
                self.selected_videos.append(self.data_files[i])      
        # Get list of all files 
        if self.use_current_video_checkbox.isChecked():
            self.use_current_video = True

        self.show_step_3()

    def show_step_3(self):
        print("Path set to {}".format(self.output_folder_path))
        print("Model set to {}".format(self.model_files))
        print("Use current video: {}".format(self.use_current_video))
        print("Selected videos: {}".format(self.selected_videos))
        if self.use_current_video:
            self.show_refinement_options()
        else:
            self.train_model()

    def show_refinement_options(self):

        self.clear_window()
        self.update_window_title("Step 3: Keypoints refinement")

        

    def train_model(self):
        # Get the selected videos
        print("Training using the following videos: {} ".format(self.selected_videos))

        """
        # Create a new thread to train the model
        self.train_thread = TrainThread(self.model_files, self.selected_videos, self.output_folder_path, self.gui)
        self.train_thread.start()
        self.train_thread.finished.connect(self.show_refinement_options)
        """

