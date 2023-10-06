"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import os
import shutil
from glob import glob

import cv2
import numpy as np
import pyqtgraph as pg
import torch
from matplotlib import cm
from qtpy import QtCore, QtGui
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
)
from torch.nn import functional as F

from .. import utils
from ..gui import help_windows, io
from . import model_loader, transforms

"""
Single workflow for re-training the model or fine-tuning the model with new data.
Keypoints correction feature for new mouse videos
"""

BODYPARTS = [
    "eye(back)",
    "eye(bottom)",
    "eye(front)",
    "eye(top)",
    "lowerlip",
    "mouth",
    "nose(bottom)",
    "nose(r)",
    "nose(tip)",
    "nose(top)",
    "nosebridge",
    "paw",
    "whisker(I)",  # "whisker(c1)",
    "whisker(III)",  # "whisker(d2)",
    "whisker(II)",  # "whisker(d1)",
]


class ModelTrainingPopup(QDialog):
    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.output_folder_path = None
        self.output_model_name = None
        self.model_files = None
        self.data_files = None
        self.selected_videos = None
        self.use_current_video = False
        self.num_video_frames = []
        self.random_frames_ind = []
        self.difficult_frames_idx = None
        self.easy_frames_idx = None
        self.pose_data = []
        self.all_frames = []
        self.bbox = []
        self.current_video_idx = 0
        # Training parameters
        self.batch_size = 1
        self.epochs = 36
        self.learning_rate = 1e-4
        self.weight_decay = 0
        self.bodyparts = BODYPARTS
        self.brushes, self.colors = self.get_brushes(self.bodyparts)

        self.setWindowTitle("Step 1: Set folders")
        self.setStyleSheet("QDialog {background: 'black';}")

        self.verticalLayout = QVBoxLayout(self)
        # Set window size that is adjusted to the size of the window
        self.window_max_size = QtGui.QGuiApplication.primaryScreen().availableGeometry()

        self.show_choose_folder()
        self.setLayout(self.verticalLayout)

        self.show()

    def update_window_size(self, frac=0.5, aspect_ratio=1.0):
        # Set the size of the window to be a fraction of the screen size using the aspect ratio
        self.resize(
            int(np.floor(self.window_max_size.width() * frac)),
            int(np.floor(self.window_max_size.height() * frac * aspect_ratio)),
        )
        # center window on screen
        centerPoint = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        qtRectangle = self.frameGeometry()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def clear_window(self):
        # Clear the window
        for i in reversed(range(self.verticalLayout.count())):
            self.verticalLayout.itemAt(i).widget().setParent(None)

    def update_window_title(self, title=None):
        if title is None:
            self.setWindowTitle(
                "Keypoints refinement: frame {}/{}".format(
                    self.current_frame + 1, sum(self.num_video_frames)
                )
            )
        else:
            self.setWindowTitle(title)

    #### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Step 1: Choose folder ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ####

    def show_choose_folder(self):
        """
        Ask user to set path to output folder where refined keypoints and trained model will be saved
        """
        self.update_window_size(0.25, aspect_ratio=0.8)

        # Add a QLineEdit widget to the layout and a browse button to set the path to the output folder
        self.output_folder_path = QLabel(self)
        self.output_folder_path.setText("Output folder path")
        self.output_folder_path.setStyleSheet("QLabel {color: 'white';}")
        self.verticalLayout.addWidget(self.output_folder_path)

        # Create a QGroupbox widget to hold the browse button and the QLineEdit widget with a horizontal layout
        self.output_folder_groupbox = QGroupBox(self)
        self.output_folder_groupbox.setLayout(QHBoxLayout())

        self.output_folder_path_box = QLineEdit(self)
        self.output_folder_path_box.setStyleSheet("QLineEdit {color: 'black';}")
        self.output_folder_path_box.setText("...")
        self.output_folder_groupbox.layout().addWidget(self.output_folder_path_box)

        self.output_folder_path_button = QPushButton("Browse", self)
        self.output_folder_path_button.clicked.connect(
            lambda clicked: self.set_output_folder_path()
        )
        self.output_folder_groupbox.layout().addWidget(self.output_folder_path_button)

        self.verticalLayout.addWidget(self.output_folder_groupbox)

        # Add a next button to the layout and connect it to the next step
        # Create a QGroupbox widget to hold the next button
        self.next_button_groupbox = QGroupBox(self)
        self.next_button_groupbox.setLayout(QHBoxLayout())
        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(
            lambda clicked: self.show_choose_training_files()
        )
        # Align the button to the right of the layout
        self.next_button_groupbox.layout().addWidget(
            self.next_button, alignment=QtCore.Qt.AlignRight
        )
        self.verticalLayout.addWidget(self.next_button_groupbox)

    def set_output_folder_path(self):
        path = io.get_folder_path(parent=self.gui)
        self.output_folder_path_box.setText(path)

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Step 2: Choose training files ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    def show_choose_training_files(self):
        self.output_folder_path = self.output_folder_path_box.text()
        # Check if path exists
        if not os.path.exists(self.output_folder_path):
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setText("Please set a valid output folder path first")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return

        self.clear_window()
        self.update_window_title("Step 2: Select training data")

        # Check if the selected output folder path contains a model file (*.pt) and data files (*.npy)
        self.model_files = glob(os.path.join(model_loader.get_models_dir(), "*.pt"))
        self.data_files = model_loader.get_model_files()
        if len(self.model_files) == 0:
            # If no model file exists then copy the default model file from the package to the output folder
            print("No model file found in the selected output folder")
            print("Copying default model file to the selected output folder")
            model_state_path = model_loader.get_basemodel_state_path()
            shutil.copy(model_state_path, self.output_folder_path)
            self.model_files = glob(os.path.join(self.output_folder_path, "*.pt"))

        # Add a QGroupbox widget to hold a qlabel and dropdown menu
        self.model_groupbox = QGroupBox(self)
        self.model_groupbox.setLayout(QHBoxLayout())

        self.model_label = QLabel(self)
        self.model_label.setText("Initial model:")
        self.model_label.setStyleSheet(
            "QLabel {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.model_groupbox.layout().addWidget(self.model_label)

        self.model_dropdown = QComboBox(self)
        self.model_dropdown.setFixedWidth(
            int(np.floor(self.window_max_size.width() * 0.25 * 0.5))
        )
        self.model_dropdown.view().setVerticalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOn
        )
        self.model_dropdown.setMaxVisibleItems(5)
        self.model_dropdown.setStyleSheet(
            "QComboBox { combobox-popup: 0; color: 'black';}"
        )
        # Add the model files to the dropdown menu
        self.model_dropdown.addItem("Base model")
        for model_file in self.model_files:
            if os.path.basename(model_file) == "facemap_model_state.pt":
                continue
            self.model_dropdown.addItem(os.path.basename(model_file.split(".pt")[0]))

        self.model_groupbox.layout().addWidget(
            self.model_dropdown, alignment=QtCore.Qt.AlignRight
        )

        self.verticalLayout.addWidget(self.model_groupbox)

        # Add a QGroupBox widget to hold a QLabel and QLineEdit widgets for setting model name
        self.model_name_groupbox = QGroupBox(self)
        self.model_name_groupbox.setLayout(QHBoxLayout())

        self.model_name_label = QLabel(self)
        self.model_name_label.setText("Output model name:")
        self.model_name_label.setStyleSheet(
            "QLabel {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.model_name_groupbox.layout().addWidget(self.model_name_label)

        self.model_name_box = QLineEdit(self)
        self.model_name_box.setStyleSheet("QLineEdit {color: 'black';}")
        self.model_name_box.setText("refined_model")
        self.model_name_box.setFixedWidth(
            int(np.floor(self.window_max_size.width() * 0.25 * 0.5))
        )
        self.model_name_groupbox.layout().addWidget(
            self.model_name_box, alignment=QtCore.Qt.AlignRight
        )

        self.verticalLayout.addWidget(self.model_name_groupbox)

        # Add a QCheckBox widget for user to select whether to use the current video
        self.use_current_video_groupbox = QGroupBox(self)
        self.use_current_video_groupbox.setLayout(QHBoxLayout())

        use_current_video_label = QLabel(self)
        use_current_video_label.setText("Refine current video?")
        use_current_video_label.setStyleSheet(
            "QLabel {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.use_current_video_groupbox.layout().addWidget(use_current_video_label)

        # Create a radio button group asking whether to use the current video
        self.use_current_video_radio_group = QButtonGroup(self)
        self.use_current_video_radio_group.setExclusive(True)
        self.use_current_video_radio_group.buttonClicked.connect(
            lambda: self.toggle_num_frames(self.use_current_video_yes_radio.isChecked())
        )
        # Add yes and no radio buttons to the group
        self.use_current_video_yes_radio = QRadioButton("Yes", self)
        self.use_current_video_yes_radio.setStyleSheet(
            "QRadioButton {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.use_current_video_radio_group.addButton(self.use_current_video_yes_radio)
        self.use_current_video_yes_radio.setChecked(True)
        self.use_current_video_groupbox.layout().addWidget(
            self.use_current_video_yes_radio
        )
        self.use_current_video_no_radio = QRadioButton("No", self)
        self.use_current_video_no_radio.setStyleSheet(
            "QRadioButton {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.use_current_video_radio_group.addButton(self.use_current_video_no_radio)
        self.use_current_video_groupbox.layout().addWidget(
            self.use_current_video_no_radio
        )

        # Add a QLabel and QSpinBox widget for user to select the number of frames to use in the current video group
        self.get_num_frames_groupbox = QGroupBox(self)
        self.get_num_frames_groupbox.setLayout(QHBoxLayout())

        self.current_video_label = QLabel(self)
        self.current_video_label.setText("# Frames:")
        self.current_video_label.setStyleSheet("QLabel {color: 'white';}")
        self.get_num_frames_groupbox.layout().addWidget(self.current_video_label)

        self.spinbox_nframes = QSpinBox(self)
        self.spinbox_nframes.setRange(1, self.gui.cumframes[-1])
        self.spinbox_nframes.setValue(15)
        self.spinbox_nframes.setStyleSheet("QSpinBox {color: 'black';}")
        self.get_num_frames_groupbox.layout().addWidget(self.spinbox_nframes)

        self.use_current_video_groupbox.layout().addWidget(self.get_num_frames_groupbox)
        self.verticalLayout.addWidget(self.use_current_video_groupbox)

        # Add a QLineedit to determine the percentage of random frames to use
        self.random_frames_groupbox = QGroupBox(self)
        self.random_frames_groupbox.setLayout(QHBoxLayout())

        self.percent_random_frames_label = QLabel(self)
        self.percent_random_frames_label.setText("Random frames (%):")
        self.percent_random_frames_label.setStyleSheet("QLabel {color: 'white';}")
        self.random_frames_groupbox.layout().addWidget(self.percent_random_frames_label)

        self.percent_random_frames_box = QSpinBox(self)
        self.percent_random_frames_box.setRange(0, 100)
        self.percent_random_frames_box.setValue(50)
        self.percent_random_frames_box.setStyleSheet("QSpinBox {color: 'black';}")
        self.random_frames_groupbox.layout().addWidget(self.percent_random_frames_box)

        # Add a QLabel and QLineedit for difficult frames threshold
        self.difficult_frames_threshold_label = QLabel(self)
        self.difficult_frames_threshold_label.setText(
            "Difficulty threshold (percentile):"
        )
        self.difficult_frames_threshold_label.setStyleSheet("QLabel {color: 'white';}")
        self.random_frames_groupbox.layout().addWidget(
            self.difficult_frames_threshold_label
        )

        self.difficult_frames_threshold_box = QSpinBox(self)
        self.difficult_frames_threshold_box.setRange(0, 100)
        self.difficult_frames_threshold_box.setValue(95)
        self.difficult_frames_threshold_box.setStyleSheet("QSpinBox {color: 'black';}")
        self.random_frames_groupbox.layout().addWidget(
            self.difficult_frames_threshold_box
        )

        self.verticalLayout.addWidget(self.random_frames_groupbox)

        # Add a QGroupbox widget to hold checkboxes for selecting videos using the list of data files
        self.npy_files_groupbox = QGroupBox(self)
        self.npy_files_groupbox.setLayout(QVBoxLayout())

        self.use_old_data_radio_groupbox = QGroupBox(self)
        self.use_old_data_radio_groupbox.setLayout(QHBoxLayout())
        use_old_data_label = QLabel(self)
        use_old_data_label.setText("Refine saved data?")
        use_old_data_label.setStyleSheet(
            "QLabel {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.use_old_data_radio_groupbox.layout().addWidget(use_old_data_label)
        # Add a radio button group asking whether to use old data
        self.use_old_data_radio_group = QButtonGroup(self)
        self.use_old_data_radio_group.setExclusive(True)
        self.use_old_data_radio_group.buttonClicked.connect(
            lambda: self.show_data_files(self.use_old_data_yes_radio.isChecked())
        )
        # Add yes and no radio buttons to the group
        self.use_old_data_yes_radio = QRadioButton("Yes", self)
        self.use_old_data_yes_radio.setStyleSheet(
            "QRadioButton {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.use_old_data_radio_group.addButton(self.use_old_data_yes_radio)
        self.use_old_data_radio_groupbox.layout().addWidget(self.use_old_data_yes_radio)
        self.use_old_data_no_radio = QRadioButton("No", self)
        self.use_old_data_no_radio.setStyleSheet(
            "QRadioButton {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.use_old_data_no_radio.setChecked(True)
        self.use_old_data_radio_group.addButton(self.use_old_data_no_radio)
        self.use_old_data_radio_groupbox.layout().addWidget(self.use_old_data_no_radio)
        self.npy_files_groupbox.layout().addWidget(self.use_old_data_radio_groupbox)

        # Add checkboxes to scroll area for selecting data files
        form_layout = QFormLayout()
        checkbox_groupbox = QGroupBox(self)
        self.npy_files_checkboxes = []
        for i, file in enumerate(self.data_files):
            checkbox = QCheckBox(file, self)
            checkbox.setStyleSheet("QCheckBox {color: 'black';}")  #'white';}")
            form_layout.addRow(checkbox)
            self.npy_files_checkboxes.append(checkbox)
            checkbox.hide()
        checkbox_groupbox.setLayout(form_layout)
        self.checkbox_scroll_area = QScrollArea(self)
        self.checkbox_scroll_area.setStyleSheet(
            "QScrollArea {background: 'black'; color: 'black';}"
        )
        self.checkbox_scroll_area.setWidget(checkbox_groupbox)
        self.checkbox_scroll_area.setWidgetResizable(True)
        self.checkbox_scroll_area.setFixedHeight(200)
        self.checkbox_scroll_area.hide()
        # Set background color of scroll area
        p = self.checkbox_scroll_area.palette()
        p.setColor(self.checkbox_scroll_area.backgroundRole(), QColor("black"))
        self.checkbox_scroll_area.setPalette(p)

        self.npy_files_groupbox.layout().addWidget(self.checkbox_scroll_area)

        self.old_data_found_label = QLabel(self)
        self.old_data_found_label.setText("No old data found.")
        self.old_data_found_label.setStyleSheet(
            "QLabel {color: 'white'; font-size: 16;}"
        )
        self.old_data_found_label.hide()
        self.npy_files_groupbox.layout().addWidget(
            self.old_data_found_label, alignment=QtCore.Qt.AlignCenter
        )

        self.verticalLayout.addWidget(self.npy_files_groupbox)

        # Add a + button to show additional options
        self.additional_options_groupbox = QGroupBox(self)
        self.additional_options_groupbox.setLayout(QVBoxLayout())

        self.add_button_groupbox = QGroupBox(self)
        self.add_button_groupbox.setLayout(QHBoxLayout())

        self.add_button = QPushButton(self)
        self.add_button.setText("+")
        self.add_button.setFixedSize(30, 30)
        self.add_button.setStyleSheet("QPushButton {color: 'black'; font-size: 14}")
        self.add_button.clicked.connect(
            lambda clicked: self.add_training_params_to_window()
        )  # Add training parameters to the window
        self.add_button_groupbox.layout().addWidget(self.add_button)

        self.training_params_label = QLabel(self)
        self.training_params_label.setText("Show training parameters")
        self.training_params_label.setStyleSheet("QLabel {color: 'white';}")
        self.add_button_groupbox.layout().addWidget(self.training_params_label)

        self.additional_options_groupbox.layout().addWidget(self.add_button_groupbox)
        self.verticalLayout.addWidget(
            self.additional_options_groupbox, alignment=QtCore.Qt.AlignCenter
        )

        # Add a QGroupbox widget to hold two buttons for cancel and next step with a horizontal layout aligned to the right
        self.buttons_groupbox = QGroupBox(self)
        self.buttons_groupbox.setLayout(QHBoxLayout())

        # Add a help button
        self.help_button = QPushButton("Help", self)
        self.help_button.clicked.connect(self.show_step2_help)
        self.buttons_groupbox.layout().addWidget(
            self.help_button, alignment=QtCore.Qt.AlignRight
        )

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(lambda clicked: self.close())
        self.buttons_groupbox.layout().addWidget(
            self.cancel_button, alignment=QtCore.Qt.AlignRight
        )

        self.next_button = QPushButton("Next", self)
        self.next_button.clicked.connect(
            lambda clicked: self.update_user_training_options()
        )
        self.buttons_groupbox.layout().addWidget(
            self.next_button, alignment=QtCore.Qt.AlignRight
        )

        self.verticalLayout.addWidget(self.buttons_groupbox)
        
        self.update_window_size(0.4, aspect_ratio=1.5)

    def show_data_files(self, yes_selected):
        """
        Show or hide the checkboxes for selecting data files
        """
        if yes_selected:
            if len(self.npy_files_checkboxes) == 0:
                self.old_data_found_label.show()
            else:
                self.old_data_found_label.hide()
            for checkbox in self.npy_files_checkboxes:
                checkbox.show()
            self.checkbox_scroll_area.show()
        else:
            for checkbox in self.npy_files_checkboxes:
                checkbox.hide()
            self.checkbox_scroll_area.hide()
            self.old_data_found_label.hide()

    def show_step2_help(self):
        """
        Show help for step 2 of the training process involving setting parameters and selecting data files
        """
        help_windows.PoseRefinementStep2HelpWindow(self, self.window_max_size)

    def add_training_params_to_window(self):
        """
        Add training parameters to the window
        """
        if self.add_button.text() == "+":
            self.add_button.setText("-")
            self.training_params_label.setText("Hide training parameters")

            # Add a QGroupbox widget to hold a qlabel and dropdown menu
            self.training_params_groupbox = QGroupBox(self)
            self.training_params_groupbox.setLayout(QHBoxLayout())

            # Add a label and spinbox for user to select the number of epochs to train
            self.epochs_groupbox = QGroupBox(self)
            self.epochs_groupbox.setLayout(QHBoxLayout())

            self.epochs_label = QLabel(self)
            self.epochs_label.setText("Epochs:")
            self.epochs_label.setStyleSheet("QLabel {color: 'white';}")
            self.epochs_groupbox.layout().addWidget(self.epochs_label)

            self.spinbox_epochs = QSpinBox(self)
            self.spinbox_epochs.setRange(1, 200)
            self.spinbox_epochs.setValue(self.epochs)
            self.spinbox_epochs.valueChanged.connect(
                lambda value: self.update_epochs(value)
            )
            self.spinbox_epochs.setStyleSheet("QSpinBox {color: 'black';}")
            self.epochs_groupbox.layout().addWidget(self.spinbox_epochs)

            # Add a label and QLineEdit widget for user to enter the learning rate
            self.learning_rate_groupbox = QGroupBox(self)
            self.learning_rate_groupbox.setLayout(QHBoxLayout())

            self.learning_rate_label = QLabel(self)
            self.learning_rate_label.setText("Learning rate:")
            self.learning_rate_label.setStyleSheet("QLabel {color: 'white';}")
            self.learning_rate_groupbox.layout().addWidget(self.learning_rate_label)

            self.lineedit_learning_rate = QLineEdit(self)
            self.lineedit_learning_rate.setText(str(self.learning_rate))
            self.lineedit_learning_rate.textChanged.connect(
                lambda text: self.update_learning_rate(text)
            )
            self.lineedit_learning_rate.setStyleSheet("QLineEdit {color: 'black';}")
            self.learning_rate_groupbox.layout().addWidget(self.lineedit_learning_rate)

            # Add a label and QLineEdit widget for user to enter the batch size
            self.batch_size_groupbox = QGroupBox(self)
            self.batch_size_groupbox.setLayout(QHBoxLayout())

            self.batch_size_label = QLabel(self)
            self.batch_size_label.setText("Batch size:")
            self.batch_size_label.setStyleSheet("QLabel {color: 'white';}")
            self.batch_size_groupbox.layout().addWidget(self.batch_size_label)

            self.lineedit_batch_size = QLineEdit(self)
            self.lineedit_batch_size.setText(str(self.batch_size))
            self.lineedit_batch_size.textChanged.connect(
                lambda text: self.update_batch_size(text)
            )
            self.lineedit_batch_size.setStyleSheet("QLineEdit {color: 'black';}")
            self.batch_size_groupbox.layout().addWidget(self.lineedit_batch_size)

            # Add a label and QLineEdit widget for user to enter the weight decay factor
            self.weight_decay_groupbox = QGroupBox(self)
            self.weight_decay_groupbox.setLayout(QHBoxLayout())

            self.weight_decay_label = QLabel(self)
            self.weight_decay_label.setText("Weight decay:")
            self.weight_decay_label.setStyleSheet("QLabel {color: 'white';}")
            self.weight_decay_groupbox.layout().addWidget(self.weight_decay_label)

            self.lineedit_weight_decay = QLineEdit(self)
            self.lineedit_weight_decay.setText(str(self.weight_decay))
            self.lineedit_weight_decay.textChanged.connect(
                lambda text: self.update_weight_decay(text)
            )
            self.lineedit_weight_decay.setStyleSheet("QLineEdit {color: 'black';}")
            self.weight_decay_groupbox.layout().addWidget(self.lineedit_weight_decay)

            self.training_params_groupbox.layout().addWidget(self.epochs_groupbox)
            self.training_params_groupbox.layout().addWidget(
                self.learning_rate_groupbox
            )
            self.training_params_groupbox.layout().addWidget(self.batch_size_groupbox)
            self.training_params_groupbox.layout().addWidget(self.weight_decay_groupbox)

            self.additional_options_groupbox.layout().addWidget(
                self.training_params_groupbox
            )

        else:
            self.add_button.setText("+")
            self.training_params_label.setText("Show training parameters")
            self.training_params_groupbox.deleteLater()

        return

    # Update training parameters
    def update_epochs(self, value):
        """
        Update the number of epochs to train
        """
        self.epochs = value
        return

    def update_learning_rate(self, text):
        """
        Update the learning rate
        """
        self.learning_rate = float(text)
        return

    def update_batch_size(self, text):
        """
        Update the batch size
        """
        self.batch_size = int(text)
        return

    def update_weight_decay(self, text):
        """
        Update the weight decay factor
        """
        self.weight_decay = float(text)
        return

    def toggle_num_frames(self, yes_selected):
        if yes_selected:
            self.get_num_frames_groupbox.show()
            self.random_frames_groupbox.show()
        else:
            self.get_num_frames_groupbox.hide()
            self.random_frames_groupbox.hide()

    def update_user_training_options(self):
        # Get the selected model
        self.model_files = self.model_files[self.model_dropdown.currentIndex()]
        self.output_model_name = self.model_name_box.text()
        if self.output_model_name == "facemap_model_state":
            # Open a QMessageBox to warn the user that the model name is not allowed
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Icon.Warning)
            msg.setText("Output model name cannot be 'facemap_model_state'")
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.StandardButton.Ok)
            msg.exec_()
            return
        # Get the selected videos
        self.selected_videos = []
        for i, checkbox in enumerate(self.npy_files_checkboxes):
            if checkbox.isChecked():
                self.selected_videos.append(self.data_files[i])
        # Get list of all files
        if self.use_current_video_yes_radio.isChecked():
            self.use_current_video = True
            self.num_video_frames = [self.spinbox_nframes.value()]
        else:
            self.use_current_video = False
            self.num_video_frames = []

        self.show_step_3()

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Step 3: Keypoints refinement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    def show_step_3(self):
        if self.use_current_video or len(self.selected_videos) > 0:
            self.show_refinement_options()
        else:
            # Show error message
            self.error_message = QMessageBox()
            self.error_message.setIcon(QMessageBox.Icon.Critical)
            self.error_message.setText("Please select at least one video for training.")
            self.error_message.setWindowTitle("Error")
            self.error_message.exec_()
        return

    def show_refinement_options(
        self, predict_frame_index=None, additional_frames=False
    ):
        self.clear_window()
        self.update_window_title("Step 3: Refine keypoints")
        self.update_window_size(0.6, aspect_ratio=1.3)

        self.hide()

        # Select frames for prediction
        if self.use_current_video:  # Generate predictions for random frames
            if len(self.random_frames_ind) == 0:
                self.random_frames_ind = self.get_random_frames(
                    total_frames=self.gui.cumframes[-1],
                    size=min(
                        250, self.gui.cumframes[-1]
                    ),  # self.num_video_frames[self.current_video_idx],
                )
                frames_indices = self.random_frames_ind
            else:  # Use the predictions from the previous step and add new predictions
                frames_indices = predict_frame_index

            # Get the predictions for the selected frames
            output = self.generate_predictions(
                frames_indices, model_name=self.model_dropdown.currentText()
            )
            if output is None:  # User cancelled the refinement
                self.close()
                return

            pose_pred, _, bbox = output
            if self.difficult_frames_idx is None or self.easy_frames_idx is None:
                (
                    difficult_frames_idx,
                    easy_frames_idx,
                ) = self.split_frames_idx_by_category(pose_pred)
                self.difficult_frames_idx = frames_indices[difficult_frames_idx]
                self.easy_frames_idx = frames_indices[easy_frames_idx]
                num_easy_frames = int(
                    np.floor(
                        self.num_video_frames[self.current_video_idx]
                        * (float(self.percent_random_frames_box.text()) / 100)
                    )
                )  # self.num_video_frames[self.current_video_idx] // 2
                num_difficult_frames = (
                    self.num_video_frames[self.current_video_idx] - num_easy_frames
                )
                if num_difficult_frames > len(self.difficult_frames_idx):
                    num_difficult_frames = len(self.difficult_frames_idx)
                    num_easy_frames = (
                        self.num_video_frames[self.current_video_idx]
                        - num_difficult_frames
                    )
                print(
                    "Total training frames: {}, Total easy frames: {}, Total difficult frames: {}".format(
                        self.num_video_frames[self.current_video_idx],
                        num_easy_frames,
                        num_difficult_frames,
                    )
                )
                easy_frames_idx = easy_frames_idx[:num_easy_frames]
                difficult_frames_idx = difficult_frames_idx[:num_difficult_frames]
                self.easy_frames_idx = self.easy_frames_idx[num_easy_frames:]
                self.difficult_frames_idx = self.difficult_frames_idx[
                    num_difficult_frames:
                ]
                pose_pred = pose_pred[[*easy_frames_idx, *difficult_frames_idx]]
                self.random_frames_ind = self.random_frames_ind[
                    [*easy_frames_idx, *difficult_frames_idx]
                ]
                frames_indices = self.random_frames_ind
            frames_input = self.get_frames_from_indices(frames_indices)

            # Update the predictions
            if len(self.pose_data) == 0:
                self.pose_data = [pose_pred]
                self.all_frames = [frames_input]
                self.bbox = bbox
            else:
                self.pose_data.insert(0, pose_pred)
                self.all_frames.insert(0, frames_input)
                if additional_frames:
                    self.bbox.insert(0, self.gui.bbox[0])
                else:
                    self.bbox.insert(0, bbox)

            (
                self.all_frames[0],
                self.pose_data[0],
            ) = self.get_bbox_adjusted_img_and_keypoints(
                self.all_frames[0], self.pose_data[0], self.bbox[0], resize = False
            )

        if (
            len(self.selected_videos) > 0 and not additional_frames
        ):  # Concatenate data from all selected videos
            # For refining keypoints from old data
            for video in self.selected_videos:
                dat = np.load(video, allow_pickle=True).item()
                images = dat["imgs"]
                keypoints = dat["keypoints"]
                bbox = dat["bbox"]
                for i in range(len(images)):
                    self.all_frames.append(images[i])
                    self.pose_data.append(keypoints[i])
                    self.bbox.append(bbox[i])
                    self.num_video_frames.append(len(images[i]))

        self.show()

        # Plot the predictions
        self.overall_horizontal_group = QGroupBox()
        self.overall_horizontal_group.setLayout(QHBoxLayout())

        self.left_vertical_group = QGroupBox()
        self.left_vertical_group.setLayout(QVBoxLayout())

        self.frame_group = QGroupBox()
        self.frame_group.setLayout(QHBoxLayout())
        self.win = pg.GraphicsLayoutWidget()
        self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
        self.win.setObjectName("Keypoints refinement")
        self.keypoints_scatterplot = KeypointsGraph(parent=self)
        self.frame_win = KeypointsViewBox(
            scatter_item=self.keypoints_scatterplot, invertY=True, lockAspect=True
        )  # self.win.addViewBox(invertY=True)
        #self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        #self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.win.addItem(self.frame_win)
        self.frame_group.layout().addWidget(self.win)

        self.current_frame = -1

        # Add a Frame number label and slider
        self.frame_number_label = QLabel(self)
        self.frame_number_label.setText(
            "Frame: {}/{}".format(self.current_frame + 1, len(self.pose_data))
        )
        self.frame_number_label.setStyleSheet("QLabel {color: 'white'; font-size: 16}")
        self.frame_number_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_vertical_group.layout().addWidget(self.frame_number_label)

        # Add a saturation slider
        self.saturation_group = QGroupBox()
        self.saturation_group.setLayout(QHBoxLayout())
        self.saturation_label = QLabel(self)
        self.saturation_label.setText("Saturation:")
        self.saturation_label.setStyleSheet("QLabel {color: 'white'; font-size: 16}")
        self.saturation_label.setAlignment(QtCore.Qt.AlignRight)
        self.saturation_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.saturation_slider.setMinimum(0)
        self.saturation_slider.setMaximum(100)
        self.saturation_slider.setValue(100)
        self.saturation_slider.valueChanged.connect(self.update_saturation)
        self.saturation_slider.setTracking(False)
        # Add a brightness slider to saturation group
        self.brightness_label = QLabel(self)
        self.brightness_label.setText("Brightness:")
        self.brightness_label.setStyleSheet("QLabel {color: 'white'; font-size: 16}")
        self.brightness_label.setAlignment(QtCore.Qt.AlignRight)
        self.brightness_slider = QSlider(QtCore.Qt.Horizontal, self)
        self.brightness_slider.setMinimum(0)
        self.brightness_slider.setMaximum(200)
        self.brightness_slider.setValue(100)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.brightness_slider.setTracking(False)
        self.brightness_slider.setFixedWidth(
            int(np.floor(self.window_max_size.width() * 0.5 * 0.2))
        )
        self.saturation_group.layout().addWidget(self.brightness_label)
        self.saturation_group.layout().addWidget(self.brightness_slider)
        #self.saturation_group.layout().addWidget(self.saturation_label)
        #self.saturation_group.layout().addWidget(self.saturation_slider)
        self.left_vertical_group.layout().addWidget(self.saturation_group)

        # Define buttons for main window
        self.toggle_button_group = QGroupBox()
        self.toggle_button_group.setLayout(QHBoxLayout())
        # Add a help button
        self.refinement_help_button = QPushButton(self)
        self.refinement_help_button.setText("Help")
        self.refinement_help_button.clicked.connect(self.show_refinement_help)
        self.previous_button = QPushButton("Previous")
        self.previous_button.setEnabled(False)
        self.previous_button.clicked.connect(self.previous_frame)
        # Add a button for next step
        self.next_button = QPushButton("Next")
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.next_frame)
        self.toggle_button_group.layout().addWidget(self.refinement_help_button)
        self.toggle_button_group.layout().addWidget(self.previous_button)
        self.toggle_button_group.layout().addWidget(self.next_button)

        # Add a train model button and set alignment to the center
        self.train_button_group = QGroupBox()
        self.train_button_group.setLayout(QHBoxLayout())
        self.train_button = QPushButton("Train model")
        self.train_button.clicked.connect(self.train_model)
        self.train_button_group.layout().addWidget(
            self.train_button, alignment=QtCore.Qt.AlignCenter
        )

        # Position buttons
        self.left_vertical_group.layout().addWidget(self.frame_group)
        self.left_vertical_group.layout().addWidget(self.toggle_button_group)
        self.left_vertical_group.layout().addWidget(self.train_button_group)

        # Radio buttons group for selecting the bodyparts to be corrected
        self.radio_group = QGroupBox()
        self.radio_group.setLayout(QVBoxLayout())
        # Add a label for the radio buttons
        self.radio_label = QLabel("Bodyparts")
        self.radio_label.hide()
        self.radio_group.layout().addWidget(self.radio_label)
        self.radio_buttons_group = QButtonGroup()
        self.radio_buttons_group.setExclusive(True)
        self.radio_buttons_group.setObjectName("radio_buttons_group")
        self.radio_buttons = []
        for i, bodypart in enumerate(self.bodyparts):
            self.radio_buttons.append(QRadioButton(bodypart))
            # self.radio_buttons[i].hide()
            # Change color of radio button
            color = QColor(self.colors[i][0], self.colors[i][1], self.colors[i][2])
            alpha = 150
            values = "{r}, {g}, {b}, {a}".format(
                r=color.red(), g=color.green(), b=color.blue(), a=alpha
            )
            self.radio_buttons[i].setStyleSheet(
                "QRadioButton { background-color: rgba("
                + values
                + "); color: 'white'; border: 1px solid black; }"
            )
            self.radio_buttons[i].setObjectName(bodypart)
            self.radio_buttons[i].clicked.connect(self.radio_button_clicked)
            self.radio_buttons_group.addButton(self.radio_buttons[i])
            self.radio_group.layout().addWidget(self.radio_buttons[i])
            if i == 0:
                self.radio_buttons[i].setChecked(True)

        self.overall_horizontal_group.layout().addWidget(self.left_vertical_group)
        self.overall_horizontal_group.layout().addWidget(self.radio_group)

        self.verticalLayout.addWidget(self.overall_horizontal_group)

        self.next_frame()

    def update_saturation(self):
        """
        Update the saturation of the image
        """
        saturation = (
            float(self.saturation_slider.value())
            / 100
            * (self.all_frames[self.current_video_idx][self.current_frame].max())
        )
        self.img.setLevels([0, saturation])

    def update_brightness(self):
        """
        Update the brightness of the image
        """
        brightness_factor = float(self.brightness_slider.value())/100
        # change brightness
        image = self.all_frames[self.current_video_idx][self.current_frame]
        # Ensure that the brightness factor is within a valid range
        brightness_factor = max(0.0, min(brightness_factor, 2.0))
        # Adjust the brightness by multiplying with the factor
        brightened_image_array = image * brightness_factor
        # Make sure the values are still in the valid 0-1 range
        brightened_image_array = np.clip(brightened_image_array, 0.0, 1.0)
        self.img.setImage(brightened_image_array)


    def split_frames_idx_by_category(self, pose_pred_data):
        """
        Split the frames into difficult or easy frames category based on the likelihood threshold percentile
        and return the indices of the frames in each category.
        Parameters
        ----------
        pose_pred_data : ND-array
            Array containing the x,y keypoints position and likelihood values for each frame.
        Returns
        -------
        difficult_frames_idx : list
            List of indices of the frames that are difficult.
        easy_frames_idx : list
            List of indices of the frames that are not difficult.
        """
        likelihood_threshold_percentile = float(
            self.difficult_frames_threshold_box.text()
        )
        likelihood = pose_pred_data[:, :, -1].mean(axis=1)
        likelihood_threshold = np.nanpercentile(
            likelihood, likelihood_threshold_percentile
        )
        difficult_frames_idx = np.where(likelihood >= likelihood_threshold)[0]
        easy_frames_idx = np.where(likelihood < likelihood_threshold)[0]
        return difficult_frames_idx, easy_frames_idx

    def get_bbox_adjusted_img_and_keypoints(self, imgs, keypoints, bbox, resize=None):
        """
        Adjusts the images and keypoints to the bounding box
        Parameters
        ----------
        imgs : ND-array
            Images to be adjusted
        bbox : list
            Bounding box of the image
        keypoints : ND-array
            Keypoints to be adjusted
        """
        y1, y2, x1, x2 = bbox
        if x2 - x1 != y2 - y1:
            add_padding = True
        else:
            add_padding = False
        if resize is None:
            if x2 - x1 != 256 or y2 - y1 != 256:
                resize = True
            else:
                resize = False
        imgs, postpad_shape, pads = transforms.preprocess_img(
            torch.tensor(imgs[np.newaxis, ...]),
            bbox,
            add_padding,
            resize,
        )
        imgs = imgs.numpy().squeeze(0)
        # Adjust keypoints accordingly
        keypoints[:, :, 0] -= x1
        keypoints[:, :, 1] -= y1
        if add_padding:
            (
                keypoints[:, :, 0],
                keypoints[:, :, 1],
            ) = transforms.adjust_keypoints_for_padding(
                keypoints[:, :, 0],
                keypoints[:, :, 1],
                [-1 * val for val in pads],
            )
        if resize:
            (
                keypoints[:, :, 0],
                keypoints[:, :, 1],
            ) = transforms.rescale_keypoints(
                keypoints[:, :, 0],
                keypoints[:, :, 1],
                postpad_shape,
                (256, 256),
            )
        return imgs, keypoints

    def get_frames_from_indices(self, indices):
        # Pre-pocess images
        imall = np.zeros((len(indices), 1, self.gui.Ly[0], self.gui.Lx[0]))
        for i, index in enumerate(indices):
            frame = utils.get_frame(
                index, self.gui.nframes, self.gui.cumframes, self.gui.video
            )[0].squeeze()
            # Convert to grayscale
            imall[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Convert to int
            imall[i] = imall[i].astype(int)
        return imall.squeeze()

    def show_refinement_help(self):
        help_windows.RefinementHelpWindow(self, self.window_max_size)

    def generate_predictions(self, frame_indices, model_name):
        output = self.gui.process_subset_keypoints(frame_indices, model_name)
        return output

    def radio_button_clicked(self):
        # Change background color of the selected radio button to None
        for i, button in enumerate(self.radio_buttons):
            if button.isChecked():
                color = QColor(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                alpha = 0
                values = "{r}, {g}, {b}, {a}".format(
                    r=color.red(), g=color.green(), b=color.blue(), a=alpha
                )
                button.setStyleSheet(
                    "QRadioButton { background-color: rgba("
                    + values
                    + "); color: 'white'; border: 1px solid black; }"
                )
            else:
                color = QColor(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                alpha = 150
                values = "{r}, {g}, {b}, {a}".format(
                    r=color.red(), g=color.green(), b=color.blue(), a=alpha
                )
                button.setStyleSheet(
                    "QRadioButton { background-color: rgba("
                    + values
                    + "); color: 'white'; border: 0px solid black; }"
                )

    def plot_keypoints(self, frame_ind):
        # Plot the keypoints of the selected frames
        plot_pose_data = self.pose_data[self.current_video_idx][frame_ind]
        # Append pose data to list for each video_id
        x = plot_pose_data[:, 0]
        y = plot_pose_data[:, 1]
        # Add a scatter plot item to the window for each bodypart
        self.keypoints_scatterplot.setData(
            pos=np.array([x, y]).T,
            symbolBrush=self.brushes,
            symbolPen=pg.mkPen(color=(255, 255, 255)),  # self.colors,
            symbol="o",
            brush=self.brushes,
            hoverable=True,
            hoverSize=self.gui.sizeObject.height() * 0.01,
            hoverSymbol="x",
            pxMode=True,
            hoverBrush="r",
            name=self.bodyparts,
            data=self.bodyparts,
            size=self.gui.sizeObject.height() * 0.008,
        )
        self.frame_win.addItem(self.keypoints_scatterplot)

    def get_brushes(self, bodyparts):
        num_classes = len(bodyparts)
        colors = cm.get_cmap("jet")(np.linspace(0, 1.0, num_classes))
        colors *= 255
        colors = colors.astype(int)
        colors[:, -1] = 230
        brushes = [pg.mkBrush(color=c) for c in colors]
        return brushes, colors

    def previous_frame(self):
        # Go to previous frame
        self.update_frame_counter("prev")
        if (
            self.current_frame >= 0
            and self.current_frame <= self.num_video_frames[self.current_video_idx]
        ):
            self.frame_win.clear()
            self.next_button.setEnabled(True)
            selected_frame = self.all_frames[self.current_video_idx][self.current_frame]
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.plot_keypoints(self.current_frame)
            self.update_window_title()
        elif self.current_video_idx > 0:
            self.current_video_idx -= 1
            self.current_frame = self.num_video_frames[self.current_video_idx]
            self.previous_frame()
        else:
            self.previous_button.setEnabled(False)
        self.update_brightness()
        # self.update_saturation()

    def update_frame_counter(self, button):
        self.next_button.setEnabled(True)
        self.previous_button.setEnabled(True)

        if button == "prev":
            self.current_frame -= 1
        elif button == "next":
            self.current_frame += 1
        # Update the frame number label
        if self.current_video_idx == 0:
            self.frame_number_label.setText(
                "Frame: "
                + str(self.current_frame + 1)
                + "/"
                + str(sum(self.num_video_frames))
            )
        else:
            self.frame_number_label.setText(
                "Frame: "
                + str(
                    sum(self.num_video_frames[: self.current_video_idx])
                    + self.current_frame
                    + 1
                )
                + "/"
                + str(sum(self.num_video_frames))
            )
        self.keypoints_scatterplot.save_refined_data()
        if (
            self.current_video_idx == len(self.num_video_frames) - 1
            and self.current_frame == self.num_video_frames[self.current_video_idx] - 1
        ):
            self.next_button.setEnabled(False)
        if self.current_video_idx == 0 and self.current_frame == 0:
            self.previous_button.setEnabled(False)

    def next_frame(self):
        self.update_frame_counter("next")
        # Display next frame
        if self.current_frame < self.num_video_frames[self.current_video_idx]:
            self.frame_win.clear()
            selected_frame = self.all_frames[self.current_video_idx][self.current_frame]
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.update_window_title()
            self.plot_keypoints(self.current_frame)
            self.next_button.setText("Next")
        elif self.current_frame >= self.num_video_frames[
            self.current_video_idx
        ] and self.current_video_idx < len(self.num_video_frames):
            self.current_video_idx += 1
            self.current_frame = -1
            self.next_frame()
        else:
            self.next_button.setEnabled(False)
        #self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        #self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.update_brightness()
        #self.update_saturation()
        self.win.show()

    def keyPressEvent(self, ev):
        # Add a keyPressEvent for deleting the selected keypoint using the delete key and set the value to NaN
        # If shift and 'D' are pressed, delete the selected keypoint
        if (
            ev.key() == QtCore.Qt.Key_D and ev.modifiers() == QtCore.Qt.ShiftModifier
        ) or ev.key() in (
            QtCore.Qt.Key_Backspace,
            QtCore.Qt.Key_Delete,
        ):
            self.delete_keypoint()
        else:
            return

    def delete_keypoint(self):
        # Get index of radio button that is selected
        index = self.radio_buttons_group.checkedId()
        # Get the bodypart that is selected
        bodypart = self.radio_buttons_group.button(index).text()
        # Get index of item in an array
        selected_items = np.where(np.array(self.bodyparts) == bodypart)[0]
        for i in selected_items:
            if not np.isnan(self.keypoints_scatterplot.data["pos"][i]).any():
                self.keypoints_scatterplot.data["pos"][i] = np.nan
                # Update radio buttons
                if i < len(self.radio_buttons) - 1:
                    self.radio_buttons[i + 1].setChecked(True)
                    self.radio_buttons[i + 1].clicked.emit(True)
                self.keypoints_scatterplot.updateGraph(dragged=True)
            else:
                return

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    def train_model(self):
        if self.use_current_video:
            self.keypoints_scatterplot.save_refined_data()
        # Combine all keypoints and image data from selected videos into one array
        # resize all to 256,256 before concatenating
        for s, set in enumerate(self.all_frames):
            self.pose_data[s][:,:,0], self.pose_data[s][:,:,1] = transforms.rescale_keypoints(self.pose_data[s][:,:,0], self.pose_data[s][:,:,1], set.shape[1:], (256, 256))
            self.all_frames[s] = transforms.resize_image(torch.from_numpy(self.all_frames[s]), resize_shape=(256, 256)).numpy()

        keypoints_data = np.concatenate(self.pose_data, axis=0)
        image_data = np.concatenate(self.all_frames, axis=0)
        keypoints_data = np.array(keypoints_data)
        image_data = np.array(image_data)
        print("Keypoints data shape: ", keypoints_data.shape)
        print("Image data shape: ", image_data.shape)

        # Get training parameters
        num_epochs = int(self.epochs)
        batch_size = int(self.batch_size)
        learning_rate = float(self.learning_rate)
        weight_decay = float(self.weight_decay)
        print("Training model with parameters:")
        print("Number of epochs:", num_epochs)
        print("Batch size:", batch_size)
        print("Learning rate:", learning_rate)
        print("Weight decay:", weight_decay)
        print("")
        self.gui.train_model(
            image_data=image_data,
            keypoints_data=keypoints_data,
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            model_name=self.model_dropdown.currentText(),
            bbox=[[0, 256, 0, 256]],
        )
        self.show_finetuned_model_predictions()

    def save_model(self):
        output_filepath = os.path.join(
            self.output_folder_path, self.output_model_name + ".pt"
        )
        self.gui.save_pose_model(output_filepath)
        self.gui.add_pose_model(output_filepath)
        self.close()

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Model Evaluation ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    def show_finetuned_model_predictions(self):
        self.clear_window()
        self.update_window_title("Final step: Evaluate model training")
        self.update_window_size(frac=0.5, aspect_ratio=1.5)

        # Adda a label to the window describing the purpose of the window
        label = QLabel(
            "Please check predicted keypoints for the following sample frames are labelled correctly for the visible bodyparts to qualitatively evaluate model training. If most of the predictions are correct, please press the 'Save model' button. If you would like to further improve the prediction, please select 'Continue training' that will return to keypoints refinement step using additional frames."
        )
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 12; color: gray;")
        self.verticalLayout.addWidget(label)

        self.visualize_model_predictions()

        # Add a set of buttons to the window asking the user to save finetuned model or continue training
        self.buttons_groupbox = QGroupBox()
        self.buttons_groupbox.setLayout(QHBoxLayout())
        self.save_model_button = QPushButton("Save model")
        self.save_model_button.clicked.connect(self.save_model)
        self.continue_training_button = QPushButton("Continue training")
        self.continue_training_button.clicked.connect(self.continue_training)
        self.buttons_groupbox.layout().addWidget(self.continue_training_button)
        self.buttons_groupbox.layout().addWidget(self.save_model_button)
        self.verticalLayout.addWidget(self.buttons_groupbox)

    def visualize_model_predictions(self):
        # Add a grid of images to the window with the predictions of the finetuned model for each image
        self.sample_predictions_groupbox = QGroupBox()
        self.sample_predictions_groupbox.setLayout(QGridLayout())
        self.sample_predictions_groupbox.setTitle("Sample predictions")

        if self.gui.cumframes[-1] >= 9:
            num_frames_to_show = 9
        else:
            num_frames_to_show = self.gui.cumframes[-1]
        # Create a grid plot 3x3 of images
        # Get random frame indices that are not in the self.random_frames list
        random_frame_indices = []
        while len(random_frame_indices) < num_frames_to_show:
            random_frame_indices = self.get_random_frames(
                self.gui.cumframes[-1],
                size=num_frames_to_show - len(random_frame_indices),
            )
            # Check if any of the random frames are from the training set, if so, remove them
            if np.any(np.isin(random_frame_indices, self.random_frames_ind)):
                random_frame_indices = np.setdiff1d(
                    random_frame_indices, self.random_frames_ind
                )

        output = self.generate_predictions(
            random_frame_indices, model_name=None
        )  # Use the finetuned model
        if output is None:
            self.close()
            return
        pose_data, _, _ = output
        imgs = self.get_frames_from_indices(random_frame_indices)

        rows = int(np.floor(np.sqrt(len(imgs))))
        cols = int(np.ceil(len(imgs) / rows))
        # Add images to the grid
        for i in range(rows):
            for j in range(cols):
                frame = imgs[i * rows + j]
                self.win = pg.GraphicsLayoutWidget()
                self.win.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)
                frame_win = self.win.addViewBox(invertY=True)
                frame_win.setAspectLocked(True)
                frame_win.addItem(
                    pg.LabelItem("Frame {}".format(random_frame_indices[i * 3 + j] + 1))
                )
                frame_win.addItem(pg.ImageItem(frame))
                # Add a keypoints scatterplot to the window
                pose_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen("r", width=2))
                x, y = pose_data[i * 3 + j][:, 0], pose_data[i * rows + j][:, 1]
                pose_scatter.setData(
                    x=x,
                    y=y,
                    size=self.gui.sizeObject.height() * 0.003,
                    symbol="o",
                    brush=self.brushes,
                    hoverable=True,
                    hoverSize=self.gui.sizeObject.height() * 0.003,
                    hoverSymbol="x",
                    pen=(0, 0, 0, 0),
                    data=self.bodyparts,
                )
                frame_win.addItem(pose_scatter)
                self.win.show()
                self.sample_predictions_groupbox.layout().addWidget(self.win, i, j)
        self.verticalLayout.addWidget(self.sample_predictions_groupbox)

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Optional steps: further training ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###

    def continue_training(self):
        self.clear_window()
        self.update_window_title("Select number of additional frames to train")
        self.update_window_size(frac=0.3, aspect_ratio=0.2)

        self.add_frames_groupbox = QGroupBox()
        self.add_frames_groupbox.setLayout(QHBoxLayout())
        self.add_frames_groupbox.setTitle("Select number of additional frames to train")
        label = QLabel("# Frames to add:")
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 12; color: white;")
        self.add_frames_groupbox.layout().addWidget(label)
        self.add_frames_spinbox = QSpinBox()
        if sum(self.num_video_frames) == 0:
            self.add_frames_spinbox.setRange(1, self.gui.nframes)
            self.add_frames_spinbox.setValue(5)
        else:
            self.add_frames_spinbox.setRange(
                1, self.gui.nframes - sum(self.num_video_frames)
            )
            self.add_frames_spinbox.setValue(5)
        self.add_frames_groupbox.layout().addWidget(self.add_frames_spinbox)
        self.verticalLayout.addWidget(self.add_frames_groupbox)

        # Add training parameters groupbox to the window
        self.verticalLayout.addWidget(
            self.additional_options_groupbox, alignment=QtCore.Qt.AlignCenter
        )

        # Add a previous and next button to the window
        self.buttons_groupbox = QGroupBox()
        self.buttons_groupbox.setLayout(QHBoxLayout())
        self.cancel_frame_addition_button = QPushButton("Previous")
        self.cancel_frame_addition_button.clicked.connect(
            self.show_finetuned_model_predictions
        )
        self.add_frames_button = QPushButton("Next")
        self.add_frames_button.clicked.connect(self.add_additional_training_frames)
        self.buttons_groupbox.layout().addWidget(self.cancel_frame_addition_button)
        self.buttons_groupbox.layout().addWidget(self.add_frames_button)
        self.verticalLayout.addWidget(self.buttons_groupbox)

    def add_additional_training_frames(self):
        old_random_frames = self.random_frames_ind
        num_easy_frames = self.add_frames_spinbox.value() // 2
        num_difficult_frames = self.add_frames_spinbox.value() - num_easy_frames
        if num_difficult_frames > len(self.difficult_frames_idx):
            num_difficult_frames = len(self.difficult_frames_idx)
            num_easy_frames = self.add_frames_spinbox.value() - num_difficult_frames
        new_random_frames = [
            *self.easy_frames_idx[:num_easy_frames],
            *self.difficult_frames_idx[:num_difficult_frames],
        ]
        self.easy_frames_idx = self.easy_frames_idx[num_easy_frames:]
        self.difficult_frames_idx = self.difficult_frames_idx[num_difficult_frames:]
        self.random_frames_ind = np.concatenate((new_random_frames, old_random_frames))
        self.num_video_frames.insert(0, self.add_frames_spinbox.value())
        # Compare the new random frames with the old random frames to find the indices of the new random frames
        new_random_frames_ind = np.setdiff1d(self.random_frames_ind, old_random_frames)
        self.show_refinement_options(new_random_frames_ind, additional_frames=True)

    def get_random_frames(self, total_frames, size):
        # Get random frames indices
        random_frames_ind = np.random.choice(
            np.arange(total_frames), size=size, replace=False
        )
        return random_frames_ind


### Keypoints viewbox containing the keypoints scatterplot ###
class KeypointsViewBox(pg.ViewBox):
    def __init__(self, parent=None, scatter_item=None, **kwds):
        pg.ViewBox.__init__(self, parent, **kwds)
        # self.setMouseMode(self.RectMode)
        self.parent = parent
        self.scatter_item = scatter_item
        if self.scatter_item is not None:
            self.addItem(scatter_item)

    # Override mouseclick event to enable clicking on the image
    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            self.scatter_item.right_click_add_keypoint(
                self.mapSceneToView(event.scenePos())
            )
        else:
            event.ignore()


### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Keypoints graph features ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###


# Following adatped from https://github.com/pyqtgraph/pyqtgraph/blob/develop/examples/CustomGraphItem.py
class KeypointsGraph(pg.GraphItem):
    def __init__(self, parent=None, **kwargs):
        self.dragPoint = None
        self.dragOffset = None
        self.textItems = []
        self.parent = parent
        pg.GraphItem.__init__(self)
        self.scatter.sigClicked.connect(self.keypoint_clicked)

    def setData(self, **kwds):
        self.text = kwds.pop("text", [])
        self.data = kwds
        if "pos" in self.data and len(kwds) == 0:
            npts = self.data["pos"].shape[0]
            self.data["data"] = np.empty(npts, dtype=[("index", str)])
            self.data["data"] = kwds["name"]
            self.data["data"]["index"] = np.arange(npts)
        self.setTexts(self.text)
        self.updateGraph()

    def setTexts(self, text):
        for i in self.textItems:
            i.scene().removeItem(i)
        self.textItems = []
        for t in text:
            item = pg.TextItem(t)
            self.textItems.append(item)
            item.setParentItem(self)

    def hover(self):
        point_hovered = np.where(self.scatter.data["hovered"])[0]
        return point_hovered

    def updateGraph(self, dragged=False):
        pg.GraphItem.setData(self, **self.data)
        for i, item in enumerate(self.textItems):
            item.setPos(*self.data["pos"][i])
        if dragged:
            keypoints_refined = self.getData()
            self.update_pose_data(keypoints_refined)

    # Write a function that tracks all the keypoints positions for all frames and save them in a file (for later use)
    def update_pose_data(self, keypoints_refined):
        # Get values for the keypoints from the GUI and save them in a file
        x_coord = keypoints_refined[:, 0]
        y_coord = keypoints_refined[:, 1]
        likelihood = self.parent.pose_data[self.parent.current_video_idx][
            self.parent.current_frame
        ][:, 2]
        self.parent.pose_data[self.parent.current_video_idx][
            self.parent.current_frame
        ] = np.column_stack((x_coord, y_coord, likelihood))
        self.save_refined_data()

    def save_refined_data(self):
        # Save refined keypoints and images to a numpy file for current displayed video only
        video_path = self.parent.gui.filenames[0][0]
        video_name = os.path.basename(video_path).split(".")[0]
        savepath = os.path.join(
            self.parent.output_folder_path,
            self.parent.output_model_name + "_Facemap_refined_data.npy",
        )
        model_loader.update_models_data_txtfile([savepath])
        # Save the data
        np.save(
            savepath,
            {
                "imgs": self.parent.all_frames,
                "keypoints": self.parent.pose_data,
                "bbox": self.parent.bbox,
                "bodyparts": self.parent.bodyparts,
                "frame_ind": self.parent.random_frames_ind,
                "video_path": video_path,
            },
        )

    def getData(self):
        return self.data["pos"]

    def mouseDragEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            if ev.isStart():
                # We are already one step into the drag.
                # Find the point(s) at the mouse cursor when the button was first
                # pressed:
                pos = ev.buttonDownPos()
                pts = self.scatter.pointsAt(pos)
                if len(pts) == 0:
                    ev.ignore()
                    return
                self.dragPoint = pts[0]
                ind = pts[0].index()  # pts[0].data()[0]
                # Create a bool array of length equal to the number of points in the scatter plot
                # and set it to False
                bool_arr = np.zeros(len(self.data["pos"]), dtype=bool)
                # Set the value of the bool array to True for the point that was clicked
                bool_arr[ind] = True
                self.dragOffset = self.data["pos"][bool_arr] - pos
            elif ev.isFinish():
                self.dragPoint = None
                return
            else:
                if self.dragPoint is None:
                    ev.ignore()
                    return

            ind = self.dragPoint.index()  # self.dragPoint.data()[0]
            # Create a bool array of length equal to the number of points in the scatter plot
            bool_arr = np.zeros(len(self.data["pos"]), dtype=bool)
            bool_arr[ind] = True
            self.data["pos"][bool_arr] = ev.pos() + self.dragOffset[0]
            self.updateGraph(dragged=True)
            ev.accept()
            bp_selected_ind = np.where(bool_arr)[0][0]
            self.keypoint_clicked(None, None, bp_selected_ind)

        else:
            ev.ignore()
            return

    def keypoint_clicked(self, obj=None, points=None, ind=None):
        # Check if right or left click was performed
        if points is not None:
            # Get the index of the clicked bodypart
            ind = points[0].index()
        # Update radio button for bodypart
        self.parent.radio_buttons[ind].setChecked(True)
        self.parent.radio_buttons[ind].clicked.emit(True)

    # Add feature for adding a keypoint to the scatterplot
    def right_click_add_keypoint(self, add_point_pos):
        """
        Use right click to add a keypoint to the scatter plot (if the keypoint is not already present)
        """
        # Get the name of the bodypart selected in the radio buttons
        bp_selected = None
        selected_bp_ind = None
        for i, bp in enumerate(self.parent.bodyparts):
            if self.parent.radio_buttons[i].isChecked():
                bp_selected = bp
                selected_bp_ind = i
                break
        # Check if position of bodypart is nan
        selected_bp_pos = self.data["pos"][selected_bp_ind]
        x, y = selected_bp_pos[0], selected_bp_pos[1]
        # If keypoint is deleted, then add it back using the user selected position
        if np.isnan(x) and np.isnan(y):
            # Get position of mouse from the mouse event
            add_x, add_y = add_point_pos.x(), add_point_pos.y()
            keypoints_refined = self.getData()
            keypoints_refined[selected_bp_ind] = [add_x, add_y]
            # Add a keypoint to the scatter plot at the clicked position to add the bodypart
            self.scatter.setData(
                pos=np.array(keypoints_refined),
                symbolBrush=self.parent.brushes,
                symbolPen=self.parent.colors,
                symbol="o",
                brush=self.parent.brushes,
                hoverable=True,
                hoverSize=self.parent.gui.sizeObject.height() * 0.006,
                hoverSymbol="x",
                pxMode=True,
                hoverBrush="r",
                name=self.parent.bodyparts,
                data=self.parent.bodyparts,
                size=self.parent.gui.sizeObject.height() * 0.006,
            )
            # Update the data
            self.updateGraph()
            # Update the pose file
            keypoints_refined = self.getData()
            self.update_pose_data(keypoints_refined)
            # Check the next bodypart in the list of radio buttons
            if selected_bp_ind < len(self.parent.bodyparts) - 1:
                self.parent.radio_buttons[selected_bp_ind + 1].setChecked(True)
                self.parent.radio_buttons[selected_bp_ind + 1].clicked.emit(True)
            else:
                self.parent.radio_buttons[0].setChecked(True)
                self.parent.radio_buttons[0].clicked.emit(True)
        else:
            return
