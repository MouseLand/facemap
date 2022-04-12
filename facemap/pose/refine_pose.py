import os
from tkinter import N
import pyqtgraph as pg
import numpy as np
from ..gui import io, help_windows
from PyQt5 import QtCore
import shutil
from . import models
from matplotlib import cm
from glob import glob
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QDialog,
    QLineEdit,
    QLabel,
    QRadioButton,
    QMessageBox,
    QSpinBox,
    QPushButton,
    QVBoxLayout,
    QComboBox,
    QGridLayout,
    QHBoxLayout,
    QVBoxLayout,
    QButtonGroup,
    QGroupBox,
    QCheckBox,
    QDesktopWidget,
)

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
    "whisker(c1)",
    "whisker(c2)",
    "whisker(d1)",
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
        self.num_random_frames = 0
        self.random_frames_ind = []
        self.pose_data = None
        self.all_frames = None
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
        self.window_max_size = QDesktopWidget().screenGeometry(-1)

        self.show_choose_folder()
        self.setLayout(self.verticalLayout)

        self.show()

    def update_window_size(self, frac=0.5, aspect_ratio=1.0):
        # Set the size of the window to be a fraction of the screen size using the aspect ratio
        self.resize(
            int(np.floor(self.window_max_size.width() * frac)),
            int(np.floor(self.window_max_size.height() * frac * aspect_ratio)),
        )

    def clear_window(self):
        # Clear the window
        for i in reversed(range(self.verticalLayout.count())):
            self.verticalLayout.itemAt(i).widget().setParent(None)

    def update_window_title(self, title=None):
        if title is None:
            self.setWindowTitle(
                "Keypoints refinement: frame {}/{}".format(
                    self.current_frame + 1, self.num_random_frames
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
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet("QLabel{ color: white}")
            msg.setText("Please set a valid output folder path first")
            msg.setWindowTitle("Warning")
            msg.exec_()
            return

        self.clear_window()
        self.update_window_title("Step 2: Select training data")
        print("Path set to {}".format(self.output_folder_path))

        # Check if the selected output folder path contains a model file (*.pt) and data files (*.npy)
        self.model_files = glob(os.path.join(models.get_models_dir(), "*.pt"))
        self.data_files = models.get_model_files()
        if len(self.model_files) == 0:
            # If no model file exists then copy the default model file from the package to the output folder
            print("No model file found in the selected output folder")
            print("Copying default model file to the selected output folder")
            model_state_path = models.get_model_state_path()
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
        # Add the model files to the dropdown menu
        for model_file in self.model_files:
            self.model_dropdown.addItem(os.path.basename(model_file))
        self.model_dropdown.setStyleSheet("QComboBox {color: 'black';}")
        self.model_dropdown.setFixedWidth(
            int(np.floor(self.window_max_size.width() * 0.25 * 0.5))
        )
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
        self.model_name_box.setText("facemap_model_state")
        self.model_name_box.setFixedWidth(
            int(np.floor(self.window_max_size.width() * 0.25 * 0.5))
        )
        self.model_name_groupbox.layout().addWidget(
            self.model_name_box, alignment=QtCore.Qt.AlignRight
        )

        self.verticalLayout.addWidget(self.model_name_groupbox)

        # Add a QGroupbox widget to hold checkboxes for selecting videos using the list of data files
        self.npy_files_groupbox = QGroupBox(self)
        self.npy_files_groupbox.setLayout(QVBoxLayout())

        self.npy_files_label = QLabel(self)
        self.npy_files_label.setText("Data files:")
        self.npy_files_label.setStyleSheet(
            "QLabel {color: 'white'; font-weight: bold; font-size: 16}"
        )
        self.npy_files_groupbox.layout().addWidget(self.npy_files_label)

        self.npy_files_checkboxes = []
        for i, file in enumerate(self.data_files):
            checkbox = QCheckBox(file, self)
            checkbox.setStyleSheet("QCheckBox {color: 'white';}")
            self.npy_files_groupbox.layout().addWidget(checkbox)
            self.npy_files_checkboxes.append(checkbox)

        self.verticalLayout.addWidget(self.npy_files_groupbox)

        # Add a QCheckBox widget for user to select whether to use the current video
        self.use_current_video_checkbox_groupbox = QGroupBox(self)
        self.use_current_video_checkbox_groupbox.setLayout(QHBoxLayout())
        self.use_current_video_checkbox = QCheckBox(self)
        self.use_current_video_checkbox.setText("Refine current video")
        self.use_current_video_checkbox.setStyleSheet("QCheckBox {color: 'white'; }")
        self.use_current_video_checkbox.stateChanged.connect(
            lambda state: self.toggle_num_frames(state)
        )
        self.use_current_video_checkbox_groupbox.layout().addWidget(
            self.use_current_video_checkbox
        )
        self.verticalLayout.addWidget(self.use_current_video_checkbox_groupbox)

        # Add a QLabel and QSpinBox widget for user to select the number of frames to use in the current video group
        self.current_video_groupbox = QGroupBox(self)
        self.current_video_groupbox.setLayout(QHBoxLayout())

        self.current_video_label = QLabel(self)
        self.current_video_label.setText("# Frames to refine:")
        self.current_video_label.setStyleSheet("QLabel {color: 'white';}")
        self.current_video_groupbox.layout().addWidget(self.current_video_label)

        self.spinbox_nframes = QSpinBox(self)
        self.spinbox_nframes.setRange(1, self.gui.cumframes[-1])
        self.spinbox_nframes.setValue(25)
        self.spinbox_nframes.setStyleSheet("QSpinBox {color: 'black';}")
        self.current_video_groupbox.layout().addWidget(self.spinbox_nframes)

        self.verticalLayout.addWidget(self.current_video_groupbox)
        self.current_video_groupbox.hide()

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

    def toggle_num_frames(self, state):
        if state == QtCore.Qt.Checked:
            self.current_video_groupbox.show()
        else:
            self.current_video_groupbox.hide()

    def update_user_training_options(self):
        # Get the selected model
        self.model_files = self.model_files[self.model_dropdown.currentIndex()]
        self.output_model_name = self.model_name_box.text()
        # Get the selected videos
        self.selected_videos = []
        for i, checkbox in enumerate(self.npy_files_checkboxes):
            if checkbox.isChecked():
                self.selected_videos.append(self.data_files[i])
        # Get list of all files
        if self.use_current_video_checkbox.isChecked():
            self.use_current_video = True
            self.num_random_frames = self.spinbox_nframes.value()
        else:
            self.use_current_video = False
            self.num_random_frames = 0

        self.show_step_3()

    ### ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Step 3: Keypoints refinement ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ###
    def show_step_3(self):
        if self.use_current_video:
            self.show_refinement_options()
        else:
            self.train_model()

    def show_refinement_options(self, predict_frame_index=None):

        self.clear_window()
        self.update_window_title("Step 3: Refine keypoints")
        self.update_window_size(0.5, aspect_ratio=1.3)

        if len(self.random_frames_ind) == 0:
            self.random_frames_ind = self.get_random_frames(
                total_frames=self.gui.cumframes[-1], size=self.num_random_frames
            )

        self.hide()
        if self.pose_data is None:
            self.pose_data, self.all_frames = self.generate_predictions(
                self.random_frames_ind
            )
        else:
            additional_pose, additional_frames = self.generate_predictions(
                predict_frame_index
            )
            self.pose_data = np.concatenate((self.pose_data, additional_pose), axis=0)
            self.all_frames = np.concatenate(
                (self.all_frames, additional_frames), axis=0
            )
        self.show()

        self.overall_horizontal_group = QGroupBox()
        self.overall_horizontal_group.setLayout(QHBoxLayout())

        self.left_vertical_group = QGroupBox()
        self.left_vertical_group.setLayout(QVBoxLayout())

        self.frame_group = QGroupBox()
        self.frame_group.setLayout(QHBoxLayout())
        self.win = pg.GraphicsLayoutWidget()
        self.win.setObjectName("Keypoints refinement")
        self.frame_win = self.win.addViewBox(invertY=True)
        self.keypoints_scatterplot = KeypointsGraph(parent=self)
        self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.frame_group.layout().addWidget(self.win)

        self.current_frame = -1

        # Add a Frame number label and slider
        self.frame_number_label = QLabel(self)
        self.frame_number_label.setText(
            "Frame: {}/{}".format(self.current_frame + 1, self.num_random_frames)
        )
        self.frame_number_label.setStyleSheet("QLabel {color: 'white'; font-size: 16}")
        self.frame_number_label.setAlignment(QtCore.Qt.AlignCenter)
        self.left_vertical_group.layout().addWidget(self.frame_number_label)

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

    def show_refinement_help(self):
        help_windows.RefinementHelpWindow(self, self.window_max_size)

    def generate_predictions(self, frame_indices):
        pred_data, im_input, _, self.bbox = self.gui.process_subset_keypoints(
            frame_indices
        )
        return pred_data, im_input

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
        print("radio button clicked")

    def plot_keypoints(self, frame_ind):
        # Plot the keypoints of the selected frames
        plot_pose_data = self.pose_data[frame_ind]
        # Append pose data to list for each video_id
        x = plot_pose_data[:, 0]
        y = plot_pose_data[:, 1]
        # Add a scatter plot item to the window for each bodypart
        self.keypoints_scatterplot.setData(
            pos=np.array([x, y]).T,
            symbolBrush=self.brushes,
            symbolPen=self.colors,
            symbol="o",
            brush=self.brushes,
            hoverable=True,
            hoverSize=25,
            hoverSymbol="x",
            pxMode=True,
            hoverBrush="r",
            name=self.bodyparts,
            data=self.bodyparts,
            size=20,
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
        if self.current_frame >= 0:
            self.frame_win.clear()
            self.next_button.setEnabled(True)
            selected_frame = self.all_frames[self.current_frame]
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.plot_keypoints(self.current_frame)
            self.update_window_title()
        else:
            self.previous_button.setEnabled(False)

    def update_frame_counter(self, button):
        if button == "prev":
            self.current_frame -= 1
        elif button == "next":
            self.current_frame += 1
        if self.current_frame == 0:
            self.previous_button.setEnabled(False)
        else:
            self.previous_button.setEnabled(True)
        if self.current_frame == self.num_random_frames - 1:
            self.next_button.setEnabled(False)
        else:
            self.next_button.setEnabled(True)
        # Update the frame number label
        self.frame_number_label.setText(
            "Frame: " + str(self.current_frame + 1) + "/" + str(self.num_random_frames)
        )
        self.keypoints_scatterplot.save_refined_data()

    def next_frame(self):
        self.update_frame_counter("next")
        # Display the next frame in list of random frames with keypoints
        if self.current_frame < self.num_random_frames:
            self.frame_win.clear()
            selected_frame = self.all_frames[self.current_frame]
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.update_window_title()
            self.plot_keypoints(self.current_frame)
            self.next_button.setText("Next")
        else:
            self.next_button.setEnabled(False)
        self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.win.show()

        # Add a keyPressEvent for deleting the selected keypoint using the delete key and set the value to NaN

    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
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
        # Get the selected videos
        # Combine all keypoints and image data from selected videos into one list
        keypoints_data = []
        image_data = []
        bbox_data = []
        for video in self.selected_videos:
            keypoints_data.append(np.load(video, allow_pickle=True).item()["keypoints"])
            image_data.append(np.load(video, allow_pickle=True).item()["imgs"])
            bbox_data.append(np.load(video, allow_pickle=True).item()["bbox"])
        if self.use_current_video_checkbox.isChecked():
            image_data.append(self.all_frames)
            keypoints_data.append(self.pose_data)
            bbox_data.append(self.bbox)
        # Convert lists to numpy arrays
        keypoints_data = np.concatenate(keypoints_data, axis=0)
        image_data = np.concatenate(image_data, axis=0)
        bbox_data = np.concatenate(bbox_data, axis=0)
        keypoints_data = np.array(keypoints_data)
        image_data = np.array(image_data)
        bbox_data = np.array(bbox_data)

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
            image_data,
            keypoints_data,
            num_epochs,
            batch_size,
            learning_rate,
            weight_decay,
        )
        self.show_finetuned_model_predictions()

    def save_model(self):
        output_filepath = os.path.join(
            self.output_folder_path, self.output_model_name + ".pt"
        )
        self.gui.save_pose_model(output_filepath)
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

        self.show_sample_predictions()

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

    def show_sample_predictions(self):
        # Add a grid of images to the window with the predictions of the finetuned model for each image
        self.sample_predictions_groupbox = QGroupBox()
        self.sample_predictions_groupbox.setLayout(QGridLayout())
        self.sample_predictions_groupbox.setTitle("Sample predictions")

        if self.gui.cumframes[-1] >= 16:
            num_frames_to_show = 16
        else:
            num_frames_to_show = self.gui.cumframes[-1]
        # Create a grid plot 4x4 of images
        # Get random frame indices that are not in the self.random_frames list
        random_frame_index = []
        while len(random_frame_index) < num_frames_to_show:
            random_frame_index = self.get_random_frames(
                self.gui.cumframes[-1],
                size=num_frames_to_show - len(random_frame_index),
            )
            # Check if any of the new random frames are already in the self.random_frames list
            if np.any(np.isin(random_frame_index, self.random_frames_ind)):
                # If so, get more random frames
                random_frame_index = np.setdiff1d(
                    random_frame_index, self.random_frames_ind
                )

        pose_data, imgs = self.generate_predictions(random_frame_index)

        rows = int(np.floor(np.sqrt(len(imgs))))
        cols = int(np.ceil(len(imgs) / rows))
        # Add images to the grid
        for i in range(rows):
            for j in range(cols):
                frame = imgs[i * rows + j]
                self.win = pg.GraphicsLayoutWidget()
                frame_win = self.win.addViewBox(invertY=True)
                frame_win.addItem(
                    pg.LabelItem("Frame {}".format(random_frame_index[i * 4 + j] + 1))
                )
                frame_win.addItem(pg.ImageItem(frame))
                # Add a keypoints scatterplot to the window
                pose_scatter = pg.ScatterPlotItem(size=10, pen=pg.mkPen("r", width=2))
                x, y = pose_data[i * 4 + j][:, 0], pose_data[i * rows + j][:, 1]
                pose_scatter.setData(
                    x=x,
                    y=y,
                    size=12,
                    symbol="o",
                    brush=self.brushes,
                    hoverable=True,
                    hoverSize=12,
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
        if self.num_random_frames == 0:
            self.add_frames_spinbox.setRange(1, self.gui.nframes)
            self.add_frames_spinbox.setValue(25)
        else:
            self.add_frames_spinbox.setRange(
                1, self.gui.nframes - self.num_random_frames
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
        total_frames_to_train = self.add_frames_spinbox.value() + self.num_random_frames
        old_random_frames = self.random_frames_ind
        while not self.num_random_frames == total_frames_to_train:
            new_random_frames = self.get_random_frames(
                self.gui.cumframes[-1], total_frames_to_train - self.num_random_frames
            )
            # Check if any of the new random frames are already in the self.random_frames list
            if np.any(np.isin(new_random_frames, self.random_frames_ind)):
                # If so, get more random frames
                new_random_frames = np.setdiff1d(
                    new_random_frames, self.random_frames_ind
                )
                self.random_frames_ind = np.concatenate(
                    (self.random_frames_ind, new_random_frames)
                )
                self.num_random_frames += len(new_random_frames)
                # Check how many more random frames are needed
            else:
                # If not, update the self.random_frames list
                self.random_frames_ind = np.concatenate(
                    (self.random_frames_ind, new_random_frames), axis=0
                )
                self.num_random_frames += len(new_random_frames)
        # Compare the new random frames with the old random frames to find the indices of the new random frames
        new_random_frames_ind = np.setdiff1d(self.random_frames_ind, old_random_frames)
        self.show_refinement_options(new_random_frames_ind)

    def get_random_frames(self, total_frames, size):
        # Get random frames indices
        random_frames_ind = np.random.choice(
            np.arange(total_frames), size=size, replace=False
        )
        return random_frames_ind

    def add_to_pose_data(self, pose_data, frame_index):
        # Get the frame index of the current frame
        frame_index = np.where(self.random_frames_ind == frame_index)[0][0]
        # Get the pose data for the current frame
        pose_data = np.concatenate((pose_data, self.pose_data[frame_index]), axis=0)
        return pose_data


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
        likelihood = self.parent.pose_data[self.parent.current_frame][:, 2]
        self.parent.pose_data[self.parent.current_frame] = np.column_stack(
            (x_coord, y_coord, likelihood)
        )
        self.save_refined_data()

    # Save refined keypoints and images to a numpy file
    def save_refined_data(self):
        keypoints = self.parent.pose_data
        video_path = self.parent.gui.filenames[0][0]
        video_name = os.path.basename(video_path).split(".")[0]
        savepath = os.path.join(
            self.parent.output_folder_path,
            video_name + "_Facemap_refined_images_landmarks.npy",
        )
        models.update_models_data_txtfile([savepath])
        # Save the data
        np.save(
            savepath,
            {
                "imgs": self.parent.all_frames,
                "keypoints": keypoints,
                "bbox": self.parent.bbox,
                "bodyparts": self.parent.bodyparts,
                "frame_ind": self.parent.random_frames_ind,
                "video_path": video_path,
            },
        )

    def getData(self):
        return self.data["pos"]

    def mousePressEvent(self, QMouseEvent):
        print("mousePressEvent", QMouseEvent.button())
        if (
            QMouseEvent.button() == QtCore.Qt.LeftButton
            and QMouseEvent.modifiers() == QtCore.Qt.ControlModifier
        ):
            QMouseEvent.accept()
            self.right_click_keypoint(QMouseEvent)
        elif QMouseEvent.button() == QtCore.Qt.RightButton:
            QMouseEvent.accept()
            self.right_click_keypoint(QMouseEvent)
        elif QMouseEvent.button() == QtCore.Qt.LeftButton:
            QMouseEvent.accept()
            super().mousePressEvent(QMouseEvent)
        else:
            QMouseEvent.ignore()

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
    def right_click_keypoint(self, mouseEvent=None):
        # Get the name of the bodypart selected in the radio buttons
        bp_selected = None
        selected_bp_ind = None
        for i, bp in enumerate(self.parent.bodyparts):
            if self.parent.radio_buttons[i].isChecked():
                bp_selected = bp
                selected_bp_ind = i
                break
        print("bodypart selected", bp_selected)
        print("bodypart index", selected_bp_ind)
        # Check if position of bodypart is nan
        print("bodypart coords", self.data["pos"][selected_bp_ind])
        selected_bp_pos = self.data["pos"][selected_bp_ind]
        x, y = selected_bp_pos[0], selected_bp_pos[1]
        # If keypoint is deleted, then add it back using the user selected position
        if np.isnan(x) and np.isnan(y):
            # Get position of mouse from the mouse event
            add_point_pos = mouseEvent.pos()
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
                hoverSize=25,
                hoverSymbol="x",
                pxMode=True,
                hoverBrush="r",
                name=self.parent.bodyparts,
                data=self.parent.bodyparts,
                size=20,
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
