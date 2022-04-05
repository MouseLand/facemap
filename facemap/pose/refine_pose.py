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
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QWidget, QLineEdit, QLabel, QDialogButtonBox,
                            QSpinBox, QPushButton, QVBoxLayout, QRadioButton, QMessageBox,
                            QHBoxLayout, QVBoxLayout, QButtonGroup, QGroupBox,
                            QListWidget, QAbstractItemView, QDesktopWidget)

"""
Single workflow for re-training the model or fine-tuning the model with new data.
Trained model overwrites any previously trained model.
"""
class ModelTrainingPopup(QDialog):
    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.output_folder_path = None

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


