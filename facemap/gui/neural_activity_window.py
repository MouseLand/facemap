import os

import matplotlib
import numpy as np
import pyqtgraph as pg
import scipy.io as sio
from matplotlib import cm
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt, QUrl
from qtpy.QtWidgets import *

from facemap import utils
from facemap.gui import guiparts, help_windows, io
from facemap.neural_prediction import neural_activity, prediction_utils


class NeuralActivityWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None, window_size=None):
        super().__init__(parent)
        self.setWindowTitle("Neural activity")
        self.parent = parent
        # Set the size of the window
        if window_size is None:
            self.sizeObject = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        else:
            self.sizeObject = window_size
        self.resize(np.floor(self.sizeObject.width() / 1.5).astype(int), np.floor(self.sizeObject.height() / 2).astype(int))
        self.center()
        # Set up the UI
        self.setup_ui()

        # Initialize variables
        self.neural_activity = neural_activity.NeuralActivity(parent=self)
        self.neural_predictions = neural_activity.NeuralActivity(parent=self)
        self.neural_data_loaded = False
        self.neural_predictions_loaded = False
        self.neural_activity_vtick = None
        self.neural_predictions_vtick = None

        self.show()
        
    def center(self):
        qr = self.frameGeometry()
        cp = QtGui.QGuiApplication.primaryScreen().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    def reset(self):
        self.neural_activity_plot.clear()
        self.neural_predictions_plot.clear()
        self.neural_activity_vtick = None
        self.neural_predictions_vtick = None
        self.neural_data_loaded = False
        self.neural_predictions_loaded = False

    def setup_ui(self):
        # Set up the splitter
        splitter = QSplitter(self)
        self.setCentralWidget(splitter)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        splitter.setSizes([100,500])

        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)
        
        # Add a logo to the button panel
        icon_path = os.path.join(os.path.dirname(__file__), "..", "mouse.png")
        logo = QtGui.QPixmap(icon_path).scaled((1352/1.5)/8, (878/2)/4, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        logoLabel = QLabel(self)
        logoLabel.setPixmap(logo)
        # align the logo to the center
        logoLabel.setAlignment(QtCore.Qt.AlignCenter)
        button_layout.addWidget(logoLabel)

        self.load_groupbox = QGroupBox("Data loader")
        self.load_groupbox.setStyleSheet(
            "QGroupBox { border: 1px solid white; border-style: outset; border-radius: 5px; color:white; padding: 5px 0px;}"
        )
        self.load_groupbox.setLayout(QGridLayout())
       
        # Load neural activity
        load_neural_activity_button = QPushButton("Load neural activity")
        load_neural_activity_button.clicked.connect(self.load_neural_data)
        self.load_groupbox.layout().addWidget(load_neural_activity_button, 0, 0)

        # Load neural predictions
        load_neural_predictions_button = QPushButton("Load neural predictions")
        load_neural_predictions_button.clicked.connect(
            lambda clicked: self.load_neural_predictions_file(None)
        )
        self.load_groupbox.layout().addWidget(load_neural_predictions_button, 1, 0)

        # Add a reset button
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self.reset)
        self.load_groupbox.layout().addWidget(reset_button, 2, 0)

        self.process_groupbox = QGroupBox("Process")
        self.process_groupbox.setStyleSheet(
            "QGroupBox { border: 1px solid white; border-style: outset; border-radius: 5px; color:white; padding: 5px 0px;}"
        )
        self.process_groupbox.setLayout(QGridLayout())
        
        # Run neural predictions
        run_neural_predictions_button = QPushButton("Run neural predictions")
        run_neural_predictions_button.clicked.connect(
            self.show_run_neural_predictions_dialog
        )
        self.process_groupbox.layout().addWidget(run_neural_predictions_button, 0, 0)
        # Add a checkable button to toggle the visibility of test data in the predictions plot
        toggle_test_data = QCheckBox("Highlight test data")
        toggle_test_data.setStyleSheet("color: gray;")
        toggle_test_data.setCheckable(True)
        toggle_test_data.setChecked(True)
        toggle_test_data.stateChanged.connect(lambda: self.toggle_testdata_display(toggle_test_data))
        self.process_groupbox.layout().addWidget(toggle_test_data, 1, 0)

        # Add the panels to the splitter
        button_layout.addWidget(self.load_groupbox)
        button_layout.addWidget(self.process_groupbox)

        splitter.addWidget(button_panel)

        # Add a plots window neural data visualization
        plots_window = pg.GraphicsLayoutWidget()
        plots_window.viewport().setAttribute(QtCore.Qt.WidgetAttribute.WA_AcceptTouchEvents, False)

        self.neural_activity_plot = plots_window.addPlot(
            name="neural_activity_plot", row=0, col=0, title="Neural activity"
        )
        self.neural_activity_plot.scene().sigMouseClicked.connect(
            self.on_click_neural_activity_plot
        )
        self.neural_activity_plot.setMouseEnabled(x=True, y=False)
        self.neural_activity_plot.setMenuEnabled(False)
        self.neural_activity_plot.hideAxis("left")
        self.neural_activity_plot.disableAutoRange()

        self.neural_predictions_plot = plots_window.addPlot(
            name="neural_predictions_plot", row=1, col=0, title="Neural predictions",
        )
        self.neural_predictions_plot.scene().sigMouseClicked.connect(
            self.on_click_neural_predictions_plot
        )
        self.neural_predictions_plot.setMouseEnabled(x=True, y=False)
        self.neural_predictions_plot.setMenuEnabled(False)
        self.neural_predictions_plot.hideAxis("left")
        self.neural_predictions_plot.disableAutoRange()
        self.neural_predictions_plot.setXLink("neural_activity_plot")
        
        self.neural_activity_vtick = None
        self.neural_predictions_vtick = None

        splitter.addWidget(plots_window)
        splitter.setSizes([self.sizeObject.width() / 4, (self.sizeObject.width() / 2)])

    # Open a QDialog to select the neural data to plot
    def load_neural_data(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Neural activity")
        dialog.setContentsMargins(10, 10, 10, 10)
        # Set size of the dialog
        dialog.setFixedWidth(np.floor(self.sizeObject.width() / 3).astype(int))
        # dialog.setFixedHeight(np.floor(self.sizeObject.height() / 2.25).astype(int))

        # Create a vertical layout for the dialog
        vbox = QtWidgets.QVBoxLayout()
        dialog.setLayout(vbox)

        # Create a grouppbox for neural activity and set a vertical layout
        neural_activity_groupbox = QtWidgets.QGroupBox("Neural activity data")
        neural_activity_groupbox.setLayout(QtWidgets.QVBoxLayout())
        neural_activity_groupbox.setStyleSheet(
            "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} "
        )

        # Add a label to the groupbox
        neural_file_groupbox = QtWidgets.QGroupBox()
        neural_file_groupbox.setLayout(QtWidgets.QHBoxLayout())
        neural_file_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        neural_data_label = QtWidgets.QLabel("Neural data:")
        neural_file_groupbox.layout().addWidget(neural_data_label)
        # Add a QLineEdit to the groupbox
        dialog.neural_data_lineedit = QtWidgets.QLineEdit()
        dialog.neural_data_lineedit.setReadOnly(True)
        neural_file_groupbox.layout().addWidget(dialog.neural_data_lineedit)
        # Add a QPushButton to the groupbox
        neural_data_button = QtWidgets.QPushButton("Browse...")
        neural_data_button.clicked.connect(
            lambda clicked: self.set_neural_data_filepath(clicked, dialog)
        )
        neural_file_groupbox.layout().addWidget(neural_data_button)
        neural_activity_groupbox.layout().addWidget(neural_file_groupbox)


        # Add a hbox for data visualization
        neural_data_vis_groupbox = QtWidgets.QGroupBox()
        neural_data_vis_groupbox.setLayout(QtWidgets.QHBoxLayout())
        neural_data_vis_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        neural_data_vis_label = QtWidgets.QLabel("Data visualization:")
        neural_data_vis_groupbox.layout().addWidget(neural_data_vis_label)
        dialog.neural_data_vis_radiobuttons = QtWidgets.QButtonGroup()
        dialog.neural_data_vis_radiobuttons.setExclusive(True)
        dialog.heatmap_button = QtWidgets.QRadioButton("Heatmap")
        dialog.heatmap_button.setChecked(True)
        dialog.trace_button = QtWidgets.QRadioButton("Traces")
        dialog.neural_data_vis_radiobuttons.addButton(dialog.heatmap_button)
        dialog.neural_data_vis_radiobuttons.addButton(dialog.trace_button)
        # Add QRadiobuttons to the hbox
        neural_data_vis_groupbox.layout().addWidget(dialog.heatmap_button)
        neural_data_vis_groupbox.layout().addWidget(dialog.trace_button)
        neural_activity_groupbox.layout().addWidget(neural_data_vis_groupbox)

        vbox.addWidget(neural_activity_groupbox)

        # Add a timestamps groupbox
        timestamps_groupbox = QtWidgets.QGroupBox("Timestamps (Optional)")
        timestamps_groupbox.setLayout(QtWidgets.QVBoxLayout())
        timestamps_groupbox.setStyleSheet(
            "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} "
        )

        # Add a groupbox for neural timestamps selection
        neural_data_timestamps_groupbox = QtWidgets.QGroupBox()
        neural_data_timestamps_groupbox.setLayout(QtWidgets.QHBoxLayout())
        neural_data_timestamps_groupbox.setStyleSheet(
            "QGroupBox { border: 0px solid gray; }"
        )
        neural_timestamps_label = QtWidgets.QLabel("Neural timestamps:")
        neural_data_timestamps_groupbox.layout().addWidget(neural_timestamps_label)
        dialog.neural_data_timestamps_lineedit = QtWidgets.QLineEdit()
        neural_data_timestamps_groupbox.layout().addWidget(
            dialog.neural_data_timestamps_lineedit
        )
        neural_timestamps_browse_button = QtWidgets.QPushButton("Browse...")
        neural_timestamps_browse_button.clicked.connect(
            lambda clicked: self.set_neural_timestamps_filepath(clicked, dialog)
        )
        neural_data_timestamps_groupbox.layout().addWidget(
            neural_timestamps_browse_button
        )
        timestamps_groupbox.layout().addWidget(neural_data_timestamps_groupbox)


        # Add a groupbpx for behav timestamps selection
        behav_data_timestamps_groupbox = QtWidgets.QGroupBox()
        behav_data_timestamps_groupbox.setLayout(QtWidgets.QHBoxLayout())
        behav_data_timestamps_groupbox.setStyleSheet(
            "QGroupBox { border: 0px solid gray; }"
        )
        behav_timestamps_label = QtWidgets.QLabel("Behavior timestamps:")
        behav_data_timestamps_groupbox.layout().addWidget(behav_timestamps_label)
        dialog.behav_data_timestamps_qlineedit = QtWidgets.QLineEdit()
        dialog.behav_data_timestamps_qlineedit.setReadOnly(True)
        behav_data_timestamps_groupbox.layout().addWidget(
            dialog.behav_data_timestamps_qlineedit
        )
        behav_timestamps_browse_button = QtWidgets.QPushButton("Browse...")
        behav_timestamps_browse_button.clicked.connect(
            lambda clicked: self.set_behav_timestamps_filepath(clicked, dialog)
        )
        behav_data_timestamps_groupbox.layout().addWidget(
            behav_timestamps_browse_button
        )

        timestamps_groupbox.layout().addWidget(behav_data_timestamps_groupbox)

        vbox.addWidget(timestamps_groupbox)

        # Add a hbox for cancel and done buttons
        neural_data_buttons_hbox = QtWidgets.QHBoxLayout()
        # Add a cancel button
        neural_data_cancel_button = QtWidgets.QPushButton("Cancel")
        neural_data_cancel_button.clicked.connect(dialog.reject)
        neural_data_buttons_hbox.addWidget(neural_data_cancel_button)
        # Add a help button 
        neural_data_help_button = QtWidgets.QPushButton("Help")
        neural_data_help_button.clicked.connect(lambda clicked: self.load_neural_data_help_clicked(clicked, dialog))
        neural_data_buttons_hbox.addWidget(neural_data_help_button)
        # Add a done button
        neural_data_done_button = QtWidgets.QPushButton("Done")
        neural_data_done_button.clicked.connect(
            lambda clicked: self.neural_data_done_clicked(clicked, dialog)
        )
        neural_data_buttons_hbox.addWidget(neural_data_done_button)
        vbox.addLayout(neural_data_buttons_hbox)

        dialog.exec_()

    def load_neural_predictions_file(self, neural_predictions_filepath=None):
        #Load neural predictions file.
        if neural_predictions_filepath is None:
            neural_predictions_filepath = io.load_npy_file(self.parent, allow_mat=False)
        dat = np.load(neural_predictions_filepath, allow_pickle=True).item()
        # Check if the file is a dictionary
        if isinstance(dat, dict):
            try:
                self.neural_predictions.data = dat["predictions"]
                extent = dat["plot_extent"]
                self.neural_predictions.data_viz_method = "heatmap"
                self.neural_predictions_loaded = True
                self.plot_neural_predictions(extent=dat["plot_extent"])
                self.highlight_test_data(dat["test_indices"], extent=extent)
                print("Variance explained (test data): {}".format(dat["variance_explained"]))
            except Exception as e:
                print("error", e)
                # Show error message
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                msg.setText("Invalid neural predictions file.")
                msg.setInformativeText(
                    "The selected file is not a valid neural predictions file."
                )
                msg.setWindowTitle("Error")
                msg.exec_()

    def show_run_neural_predictions_dialog(self):
        dialog = QtWidgets.QDialog()
        dialog.setWindowTitle("Neural activity")
        dialog.setContentsMargins(10, 10, 10, 10)
        # Set size of the dialog
        dialog.setFixedWidth(np.floor(self.sizeObject.width() / 2.5).astype(int))
        # dialog.setFixedHeight(np.floor(self.sizeObject.height() / 2.25).astype(int))

        # Create a vertical layout for the dialog
        vbox = QtWidgets.QVBoxLayout()
        dialog.setLayout(vbox)

        # Add a groupbox for neural data prediction using keypoints
        neural_data_prediction_groupbox = QtWidgets.QGroupBox()
        neural_data_prediction_groupbox.setLayout(QtWidgets.QVBoxLayout())
        neural_data_prediction_groupbox.setStyleSheet(
            "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} "
        )
        neural_data_prediction_groupbox.setTitle("Neural data prediction")

        # Add a groupbox for selecting input data for prediction
        input_data_groupbox = QtWidgets.QGroupBox()
        input_data_groupbox.setLayout(QtWidgets.QHBoxLayout())
        input_data_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        input_data_label = QtWidgets.QLabel("Input data:")
        input_data_groupbox.layout().addWidget(input_data_label)
        # Create two radio buttons named keypoints and svd to ask the user to select the input data
        # for prediction
        input_data_radio_button_group = QtWidgets.QButtonGroup()
        input_data_radio_button_group.setExclusive(True)
        dialog.input_data_keypoints_radio_button = QtWidgets.QRadioButton("Keypoints")
        dialog.input_data_keypoints_radio_button.setChecked(True)
        dialog.input_data_motsvd_radio_button = QtWidgets.QRadioButton("motSVD")
        dialog.input_data_movsvd_radio_button = QtWidgets.QRadioButton("movSVD")
        input_data_radio_button_group.addButton(
            dialog.input_data_keypoints_radio_button
        )
        input_data_radio_button_group.addButton(dialog.input_data_motsvd_radio_button)
        input_data_radio_button_group.addButton(dialog.input_data_movsvd_radio_button)
        input_data_groupbox.layout().addWidget(dialog.input_data_keypoints_radio_button)
        input_data_groupbox.layout().addWidget(dialog.input_data_motsvd_radio_button)
        input_data_groupbox.layout().addWidget(dialog.input_data_movsvd_radio_button)
        input_data_radio_button_group.buttonToggled.connect(
            lambda: self.update_hyperparameter_box(dialog)
        )
        neural_data_prediction_groupbox.layout().addWidget(input_data_groupbox)

        # Add a groupbox for selecting type of output: neural pcs or activity
        output_type_groupbox = QtWidgets.QGroupBox()
        output_type_groupbox.setLayout(QtWidgets.QHBoxLayout())
        output_type_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        output_type_label = QtWidgets.QLabel("Predict neural PCs:")
        output_type_groupbox.layout().addWidget(output_type_label)
        # Create two radio buttons named yes and no to ask the user to select the type of output
        output_type_radio_button_group = QtWidgets.QButtonGroup()
        output_type_radio_button_group.setExclusive(True)
        dialog.neural_pcs_yes_radio_button = QtWidgets.QRadioButton("Yes")
        dialog.neural_pcs_yes_radio_button.setChecked(True)
        dialog.neural_pcs_no_radio_button = QtWidgets.QRadioButton("No")
        output_type_radio_button_group.addButton(dialog.neural_pcs_yes_radio_button)
        output_type_radio_button_group.addButton(dialog.neural_pcs_no_radio_button)
        output_type_groupbox.layout().addWidget(dialog.neural_pcs_yes_radio_button)
        output_type_groupbox.layout().addWidget(dialog.neural_pcs_no_radio_button)
        neural_data_prediction_groupbox.layout().addWidget(output_type_groupbox)

        vbox.addWidget(neural_data_prediction_groupbox)

        # Add a groupbox for setting training hyperparameters for neural data prediction
        dialog.neural_model_hyperparameters_groupbox = QtWidgets.QGroupBox()
        dialog.neural_model_hyperparameters_groupbox.setLayout(QtWidgets.QHBoxLayout())
        dialog.neural_model_hyperparameters_groupbox.setStyleSheet(
            "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 1em;} "
        )
        dialog.neural_model_hyperparameters_groupbox.setTitle(
            "Training hyperparameters"
        )
        learning_rate_label = QtWidgets.QLabel("Learning rate:")
        dialog.learning_rate_line_edit = QtWidgets.QLineEdit()
        dialog.learning_rate_line_edit.setText("0.001")
        weight_decay_label = QtWidgets.QLabel("Weight decay:")
        dialog.weight_decay_line_edit = QtWidgets.QLineEdit()
        dialog.weight_decay_line_edit.setText("0.0001")
        n_epochs_label = QtWidgets.QLabel("# Epochs:")
        dialog.n_epochs_line_edit = QtWidgets.QLineEdit()
        dialog.n_epochs_line_edit.setText("300")
        num_neurons_label = QtWidgets.QLabel("# Neurons:")
        dialog.num_neurons_line_edit = QtWidgets.QLineEdit()
        dialog.num_neurons_line_edit.setText("100")
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            learning_rate_label
        )
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            dialog.learning_rate_line_edit
        )
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            weight_decay_label
        )
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            dialog.weight_decay_line_edit
        )
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(n_epochs_label)
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            dialog.n_epochs_line_edit
        )
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            num_neurons_label
        )
        dialog.neural_model_hyperparameters_groupbox.layout().addWidget(
            dialog.num_neurons_line_edit
        )
        vbox.addWidget(dialog.neural_model_hyperparameters_groupbox)

        # Add a groupbox for setting hyperparameters for linear regression
        dialog.linear_regression_hyperparameters_groupbox = QtWidgets.QGroupBox()
        dialog.linear_regression_hyperparameters_groupbox.setLayout(
            QtWidgets.QHBoxLayout()
        )
        dialog.linear_regression_hyperparameters_groupbox.setStyleSheet(
            "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 1em;} "
        )
        dialog.linear_regression_hyperparameters_groupbox.setTitle(
            "Linear regression hyperparameters"
        )
        lambda_label = QtWidgets.QLabel("Lambda:")
        dialog.lambda_line_edit = QtWidgets.QLineEdit()
        dialog.lambda_line_edit.setText("0")
        dialog.linear_regression_hyperparameters_groupbox.layout().addWidget(
            lambda_label
        )
        dialog.linear_regression_hyperparameters_groupbox.layout().addWidget(
            dialog.lambda_line_edit
        )
        tbin_label = QtWidgets.QLabel("binsize:")
        dialog.tbin_spinbox = QtWidgets.QSpinBox()
        dialog.tbin_spinbox.setMinimum(0)
        dialog.tbin_spinbox.setMaximum(100)
        dialog.tbin_spinbox.setValue(0)
        dialog.linear_regression_hyperparameters_groupbox.layout().addWidget(tbin_label)
        dialog.linear_regression_hyperparameters_groupbox.layout().addWidget(
            dialog.tbin_spinbox
        )
        dialog.linear_regression_hyperparameters_groupbox.hide()
        vbox.addWidget(dialog.linear_regression_hyperparameters_groupbox)

        # Add a groupbox for saving the neural predictions
        save_neural_predictions_groupbox = QtWidgets.QGroupBox()
        save_neural_predictions_groupbox.setLayout(QtWidgets.QVBoxLayout())
        save_neural_predictions_groupbox.setStyleSheet(
            "QGroupBox {border: 1px solid gray; border-radius: 9px; margin-top: 0.5em;} "
        )
        save_neural_predictions_groupbox.setTitle("Output")

        # Add a groupbox for asking whether to save the neural predictions
        save_output_groupbox = QtWidgets.QGroupBox()
        save_output_groupbox.setLayout(QtWidgets.QHBoxLayout())
        save_output_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        save_output_label = QtWidgets.QLabel("Save output:")
        save_output_groupbox.layout().addWidget(save_output_label)
        # Create two radio buttons named yes and no to ask the user to select whether to save the
        # neural predictions
        save_output_radio_button_group = QtWidgets.QButtonGroup()
        save_output_radio_button_group.setExclusive(True)
        dialog.save_output_yes_radio_button = QtWidgets.QRadioButton("Yes")
        dialog.save_output_yes_radio_button.setChecked(True)
        dialog.save_output_no_radio_button = QtWidgets.QRadioButton("No")
        save_output_radio_button_group.addButton(dialog.save_output_yes_radio_button)
        save_output_radio_button_group.addButton(dialog.save_output_no_radio_button)
        save_output_groupbox.layout().addWidget(dialog.save_output_yes_radio_button)
        save_output_groupbox.layout().addWidget(dialog.save_output_no_radio_button)
        save_neural_predictions_groupbox.layout().addWidget(save_output_groupbox)

        # Add a groupbox for selecting the output file path
        output_file_path_groupbox = QtWidgets.QGroupBox()
        output_file_path_groupbox.setLayout(QtWidgets.QHBoxLayout())
        output_file_path_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        output_file_path_label = QtWidgets.QLabel("Output file path:")
        output_file_path_groupbox.layout().addWidget(output_file_path_label)
        # Create a line edit to ask the user to enter the output file path
        dialog.output_file_path_line_edit = QtWidgets.QLineEdit()
        dialog.output_file_path_line_edit.setText(self.parent.save_path)
        output_file_path_groupbox.layout().addWidget(dialog.output_file_path_line_edit)
        # Create a button to ask the user to select the output file path
        output_file_path_button = QtWidgets.QPushButton("Browse")
        output_file_path_button.clicked.connect(
            lambda clicked: self.output_file_path_button_clicked(
                clicked, dialog.output_file_path_line_edit
            )
        )
        output_file_path_groupbox.layout().addWidget(output_file_path_button)
        save_neural_predictions_groupbox.layout().addWidget(output_file_path_groupbox)

        # Add a groupbox for selecting the output file name
        output_filename_groupbox = QtWidgets.QGroupBox()
        output_filename_groupbox.setLayout(QtWidgets.QHBoxLayout())
        output_filename_groupbox.setStyleSheet("QGroupBox { border: 0px solid gray; }")
        output_filename_label = QtWidgets.QLabel("Output filename:")
        output_filename_groupbox.layout().addWidget(output_filename_label)
        # Create a line edit to ask the user to enter the output filename
        dialog.output_filename_line_edit = QtWidgets.QLineEdit()
        dialog.output_filename_line_edit.setText("neural_predictions")
        output_filename_groupbox.layout().addWidget(dialog.output_filename_line_edit)
        save_neural_predictions_groupbox.layout().addWidget(output_filename_groupbox)

        vbox.addWidget(save_neural_predictions_groupbox)

        # Add a hbox for cancel and run buttons
        neural_data_buttons_hbox = QtWidgets.QHBoxLayout()
        # Add a cancel button
        neural_data_cancel_button = QtWidgets.QPushButton("Cancel")
        neural_data_cancel_button.clicked.connect(dialog.reject)
        neural_data_buttons_hbox.addWidget(neural_data_cancel_button)
        # Add a help button to open the help page for model training
        neural_data_help_button = QtWidgets.QPushButton("Help")
        neural_data_help_button.clicked.connect(
            lambda clicked: self.neural_data_help_button_clicked(clicked, dialog)
        )
        neural_data_buttons_hbox.addWidget(neural_data_help_button)
        # Add a run button
        run_predictions_button = QtWidgets.QPushButton("Run")
        run_predictions_button.clicked.connect(
            lambda clicked: self.run_neural_predictions(clicked, dialog)
        )
        run_predictions_button.setDefault(True)
        neural_data_buttons_hbox.addWidget(run_predictions_button)
        vbox.addLayout(neural_data_buttons_hbox)

        dialog.exec_()

    def update_hyperparameter_box(self, dialog):
        #Update the hyperparameter box when the user changes the input/model type.
        # Hide the training hyperparameters box if keypoints is not selected
        if dialog.input_data_keypoints_radio_button.isChecked():
            dialog.neural_model_hyperparameters_groupbox.show()
            dialog.linear_regression_hyperparameters_groupbox.hide()
        else:
            dialog.neural_model_hyperparameters_groupbox.hide()
            dialog.linear_regression_hyperparameters_groupbox.show()

    def load_neural_data_help_clicked(self, clicked, dialog):
        help_windows.LoadNeuralDataHelp(parent=dialog, window_size=self.sizeObject)

    def neural_data_help_button_clicked(self, clicked, dialog):
        help_windows.NeuralModelTrainingWindow(
            parent=dialog, window_size=self.sizeObject
        )

    def neural_data_done_clicked(self, clicked, dialog):
        neural_data_filepath = dialog.neural_data_lineedit.text()
        if not os.path.isfile(neural_data_filepath):
            QtWidgets.QMessageBox.warning(
                dialog,
                "Neural data file not found",
                "The neural data file could not be found. Please select a valid file.",
            )
            return
        else:
            neural_data = np.load(neural_data_filepath)
        if dialog.heatmap_button.isChecked():
            data_viz_method = "heatmap"
        else:
            data_viz_method = "lineplot"
        try:
            self.behavior_timestamps = np.load(
                dialog.behav_data_timestamps_qlineedit.text()
            )
        except:
            self.behavior_timestamps = np.arange(0, self.parent.nframes)
        try:
            self.neural_timestamps = np.load(
                dialog.neural_data_timestamps_lineedit.text()
            )
        except:
            self.neural_timestamps = np.arange(0, neural_data.shape[1])
        self.set_neural_data(
            neural_data_filepath,
            data_viz_method,
            self.neural_timestamps,
            self.behavior_timestamps,
        )
        dialog.accept()

    def run_neural_predictions(self, clicked, dialog):
        # Run neural predictions

        if self.neural_activity.data is None:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            msg.setText("Neural activity not loaded")
            msg.setInformativeText(
                "Please load neural activity data before running neural predictions"
            )
            msg.setWindowTitle("Error")
            msg.exec_()
            dialog.accept()
            return

        if dialog.input_data_keypoints_radio_button.isChecked():
            # Check if keypoints are loaded
            if self.parent.poseFilepath[0] is None:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                msg.setText("Keypoints not loaded")
                msg.setInformativeText(
                    "Please load keypoints before running neural predictions"
                )
                msg.setWindowTitle("Error")
                msg.exec_()
                return
            # keypoints = keypoints_utils.get_normalized_keypoints(self.poseFilepath[0])
            keypoints = utils.get_keypoints_for_neuralpred(self.parent.poseFilepath[0])
            # If the number of timestamps is not equal to the number of frames, then interpolate
            if len(self.behavior_timestamps) != self.parent.nframes:
                keypoints = keypoints[
                    np.linspace(
                        0, len(keypoints) - 1, len(self.behavior_timestamps)
                    ).astype(int)
                ]

            if dialog.neural_pcs_yes_radio_button.isChecked():
                neural_target, Vt = prediction_utils.get_neural_pcs(
                    self.neural_activity.data.copy()
                )
                print("Neural PCs shape: ", neural_target.shape)
                print("Vt shape: ", Vt.shape)
            else:
                neural_target = self.neural_activity.data.T.copy()
            print("Neural target shape: ", neural_target.shape)
            print("Keypoints shape: ", keypoints.shape)
            (
                varexp,
                varexp_neurons,
                _,
                _,
                test_indices,
                _,
                model,
            ) = prediction_utils.get_keypoints_to_neural_varexp(
                keypoints,
                neural_target,
                self.behavior_timestamps,
                self.neural_timestamps,
                verbose=True,
                learning_rate=float(dialog.learning_rate_line_edit.text()),
                n_iter=int(dialog.n_epochs_line_edit.text()),
                weight_decay=float(dialog.weight_decay_line_edit.text()),
                gui=dialog,
                GUIobject=QtWidgets,
                device=self.parent.device,
            )
            # TODO: Use num neurons input provided by the user
            predictions, _ = prediction_utils.get_trained_model_predictions(
                keypoints,
                model,
                self.behavior_timestamps,
                self.neural_timestamps,
                device=self.parent.device,
            )

            if dialog.neural_pcs_yes_radio_button.isChecked():
                print("PC varexp: {}".format(varexp * 100))
                predictions = prediction_utils.get_pca_inverse_transform(
                    predictions, Vt
                )
            else:
                print("Neural activity varexp: {}".format(varexp * 100))
                predictions = predictions.T
        else:
            try:
                if dialog.input_data_motsvd_radio_button.isChecked():
                    x_input = self.motSVDs[0]
                else:
                    x_input = self.movSVDs[0]
            except:
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                msg.setText("SVDs not loaded")
                msg.setInformativeText(
                    "Please load SVDs (proc file) before running neural predictions"
                )
                msg.setWindowTitle("Error")
                msg.exec_()
                return

            if dialog.neural_pcs_yes_radio_button.isChecked():
                neural_target, Vt = prediction_utils.get_neural_pcs(
                    self.neural_activity.data.copy()
                )
                print("Neural PCs shape: ", neural_target.shape)
                print("Vt shape: ", Vt.shape)
            else:
                neural_target = self.neural_activity.data.T.copy()
            # Use a linear model to predict neural activity from SVDs
            # Resample behavioral data to match neural data timestamps
            if len(self.behavior_timestamps) != self.parent.nframes:
                x_input = x_input[
                    np.linspace(
                        0, len(x_input) - 1, len(self.behavior_timestamps)
                    ).astype(int)
                ]
            x_input = prediction_utils.resample_data_to_neural_timestamps(
                x_input, self.behavior_timestamps, self.neural_timestamps
            )
            print(
                "Using tbin and lambda: ",
                dialog.tbin_spinbox.value(),
                dialog.lambda_line_edit.text(),
            )
            (_, varexp, test_indices, A, B, _, _) = prediction_utils.rrr_prediction(
                x_input,
                neural_target,
                tbin=dialog.tbin_spinbox.value(),
                lam=float(dialog.lambda_line_edit.text()),
                device=self.parent.device,
            )
            ranks = np.argmax(varexp)
            print(
                "Max neural activity variance explained {} using {} ranks".format(
                    max(varexp) * 100, ranks + 1
                )
            )
            predictions = (x_input @ B[:, :ranks] @ A[:, :ranks].T).T
            print("predictions shape: ", predictions.shape)
            print("neural activity shape: ", self.neural_activity.data.shape)
            # Split test indices where there is a gap of more than 1 and convert each split to a list
            test_indices = np.split(
                test_indices, np.where(np.diff(test_indices) != 1)[0] + 1
            )
            test_indices = [test_indices[i].tolist() for i in range(len(test_indices))]

            # test_indices = [np.concatenate((list(test_indices[i]), list(test_indices[i+1]))) for i in range(0, len(test_indices), 2)]
            print("Test indices length: ", len(test_indices))

        # Plot neural activity predictions
        self.set_neural_prediction_data(dialog, predictions, test_indices)

        # Save neural predictions
        if dialog.save_output_yes_radio_button.isChecked():
            save_data_dict = {
                "predictions": predictions,
                "test_indices": test_indices,
                "variance_explained": varexp,
                "plot_extent": np.array(
                    [
                        0,
                        0,
                        self.neural_activity.neural_timestamps_resampled[-1],
                        self.neural_activity.data.shape[0],
                    ]
                ),
            }
            save_dir = dialog.output_file_path_line_edit.text()
            save_filename = dialog.output_filename_line_edit.text()
            save_path = os.path.join(save_dir, save_filename)

            np.save(
                save_path + ".npy",
                save_data_dict,
            )
            print("Predictions saved to: {}".format(save_dir))

    def output_file_path_button_clicked(self, clicked, line_edit):
        # Select the output file path
        save_path = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select a directory to save the predictions"
        )
        line_edit.setText(save_path)

    def set_neural_data_filepath(self, clicked, dialog):
        # Set the neural data file
        neural_data_file = io.load_npy_file(self)
        dialog.neural_data_lineedit.setText(neural_data_file)

    def set_neural_timestamps_filepath(self, clicked, dialog):
        # Set the neural timestamps file
        neural_timestamps_file = io.load_npy_file(self)
        dialog.neural_data_timestamps_lineedit.setText(neural_timestamps_file)

    def set_behav_timestamps_filepath(self, clicked, dialog):
        # Set the behavioral data file
        behav_data_file = io.load_npy_file(self)
        dialog.behav_data_timestamps_qlineedit.setText(behav_data_file)

    def set_neural_data(
        self,
        neural_data_filepath,
        data_viz_method,
        neural_timestamps_filepath,
        behav_data_timestamps_filepath,
    ):
        # Get user settings from the dialog box to set neural activity data
        self.neural_activity.set_data(
            neural_data_filepath,
            None,
            data_viz_method,
            neural_timestamps_filepath,
            None,
            None,
            behav_data_timestamps_filepath,
            None,
            None,
        )
        self.neural_data_loaded = True
        self.plot_neural_data()

    def set_neural_prediction_data(self, dialog, data, test_indices):
        # Get user settings from the dialog box to set neural prediction data
        self.neural_predictions.set_data(
            data, None, self.neural_activity.data_viz_method
        )
        self.neural_predictions_loaded = True
        self.neural_predictions.test_data_image = None
        self.plot_neural_predictions()
        self.highlight_test_data(test_indices)
        dialog.accept()

    def highlight_test_data(self, test_indices_list, extent=None):
        # Highlight the test data in the neural predictions plot

        # Create a pyqtgraph image item with low alpha value to highlight the test data
        test_section_box = np.zeros(self.neural_predictions.data.shape)
        # Set the test section box to 1 for the test indices
        for test_idx_list in test_indices_list:
            test_section_box[:, test_idx_list[0] : test_idx_list[-1]] = 1
        self.neural_predictions.test_data_image = pg.ImageItem(
            test_section_box, opacity=0.3
        )
        self.neural_activity.test_data_image = pg.ImageItem(
            test_section_box, opacity=0.3
        )

        # Set limits of the image item
        if extent is None:
            extent = QtCore.QRect(
                0,
                0,
                self.neural_activity.neural_timestamps_resampled[-1],
                self.neural_activity.data.shape[0],
            )
        else:
            extent = QtCore.QRect(*extent)
        self.neural_predictions.test_data_image.setRect(extent)
        self.neural_activity.test_data_image.setRect(extent)

        # Change color of the image item for test sections
        c_black = matplotlib.colors.colorConverter.to_rgba("black", alpha=0)
        c_green = matplotlib.colors.colorConverter.to_rgba("green", alpha=1)
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "custom_cmap", [c_black, c_green], 255
        )
        colormap._init()
        lut = (colormap._lut * 255).view(
            np.ndarray
        )  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
        lut = lut[0:-3, :]
        self.neural_predictions.test_data_image.setLookupTable(lut)
        self.neural_activity.test_data_image.setLookupTable(lut)
        if self.neural_predictions_loaded:
            self.neural_predictions_plot.addItem(self.neural_predictions.test_data_image)
        if self.neural_data_loaded:
            self.neural_activity_plot.addItem(self.neural_activity.test_data_image)

    def toggle_testdata_display(self, button):

        # Toggle the display of test data in the neural predictions plot
        if self.neural_predictions.test_data_image is not None:
            if button.isChecked():
                self.neural_predictions.test_data_image.show()
            else:
                self.neural_predictions.test_data_image.hide()
        if self.neural_activity.test_data_image is not None and self.neural_data_loaded:
            if button.isChecked():
                self.neural_activity.test_data_image.show()

            else:
                self.neural_activity.test_data_image.hide()

    def plot_neural_data(self):
        # Clear plot
        self.neural_activity_plot.clear()

        # Note: neural data is of shape (neurons, time)
        # Create a heatmap for the neural data and add it to plot 1
        vmin = -np.percentile(self.neural_activity.data, 95)
        vmax = np.percentile(self.neural_activity.data, 95)

        if self.neural_activity.data_viz_method == "heatmap":
            self.neural_heatmap = pg.ImageItem(
                self.neural_activity.data, autoDownsample=True, levels=(vmin, vmax)
            )
            if (
                self.neural_activity.behavior_timestamps is not None
                and self.neural_activity.neural_timestamps is not None
            ):
                extent = QtCore.QRect(
                    0,
                    0,
                    self.neural_activity.neural_timestamps_resampled[-1],
                    self.neural_activity.data.shape[0],
                )
                self.neural_heatmap.setRect(extent)
            self.neural_activity_plot.addItem(self.neural_heatmap)
            colormap = cm.get_cmap("gray_r")
            colormap._init()
            lut = (colormap._lut * 255).view(
                np.ndarray
            )  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
            lut = lut[0:-3, :]
            # apply the colormap
            self.neural_heatmap.setLookupTable(lut)
        else:
            x = np.empty(
                (self.neural_activity.data.shape[0], self.neural_activity.data.shape[1])
            )
            x[:] = np.arange(self.neural_activity.data.shape[1])[np.newaxis, :]
            y = self.neural_activity.data
            neural_lineplot = guiparts.MultiLine(x, y)
            self.neural_activity_plot.addItem(neural_lineplot)
        self.neural_activity_plot.autoRange()
        # Add a vertical line to the plot to indicate the time of the current trial
        self.neural_activity_vtick = pg.InfiniteLine(
            pos=self.parent.cframe,
            angle=90,
            pen=pg.mkPen(color=(255, 0, 0), width=2, movable=True),
        )
        self.neural_activity_plot.addItem(self.neural_activity_vtick)
        self.neural_activity_plot.setXRange(0, self.neural_activity.data.shape[1])
        self.neural_activity_plot.setLimits(xMin=0, xMax=self.parent.nframes)

    def plot_neural_predictions(self, extent=None):
        # Clear plot
        self.neural_predictions_plot.clear()

        # Create a heatmap for the neural data and add it to plot 1
        vmin = -np.percentile(self.neural_predictions.data, 95)
        vmax = np.percentile(self.neural_predictions.data, 95)

        if self.neural_predictions.data_viz_method == "heatmap":
            self.neural_heatmap = pg.ImageItem(
                self.neural_predictions.data, autoDownsample=True, levels=(vmin, vmax)
            )
            if extent is not None:
                self.neural_heatmap.setRect(QtCore.QRect(*extent))
            elif (
                self.neural_activity.behavior_timestamps is not None
                and self.neural_activity.neural_timestamps is not None
            ):
                extent = QtCore.QRect(
                    0,
                    0,
                    self.neural_activity.neural_timestamps_resampled[-1],
                    self.neural_activity.data.shape[0],
                )
                self.neural_heatmap.setRect(extent)
            self.neural_predictions_plot.addItem(self.neural_heatmap)
            colormap = cm.get_cmap("gray_r")
            colormap._init()
            lut = (colormap._lut * 255).view(
                np.ndarray
            )  # Convert matplotlib colormap from 0-1 to 0 -255 for Qt
            lut = lut[0:-3, :]
            # apply the colormap
            self.neural_heatmap.setLookupTable(lut)
        else:
            x = np.empty(
                (
                    self.neural_predictions.data.shape[0],
                    self.neural_predictions.data.shape[1],
                )
            )
            x[:] = np.arange(self.neural_predictions.data.shape[1])[np.newaxis, :]
            y = self.neural_predictions.data
            neural_lineplot = guiparts.MultiLine(x, y)
            self.neural_predictions_plot.addItem(neural_lineplot)
        self.neural_predictions_plot.autoRange()
        # Add a vertical line to the plot to indicate the time of the current trial
        if self.neural_predictions_vtick is None:
            self.neural_predictions_vtick = pg.InfiniteLine(
                pos=self.parent.cframe,
                angle=90,
                pen=pg.mkPen(color=(255, 0, 0), width=2, movable=True),
            )
        self.neural_predictions_plot.addItem(self.neural_predictions_vtick)
        self.neural_predictions_plot.setXRange(0, self.neural_predictions.data.shape[1])
        self.neural_predictions_plot.setLimits(xMin=0, xMax=self.parent.nframes)

    def on_click_neural_activity_plot(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            mouse_point = self.neural_activity_plot.vb.mapSceneToView(event._scenePos)
            self.update_neural_data_vtick(mouse_point.x())

    def on_click_neural_predictions_plot(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            mouse_point = self.neural_predictions_plot.vb.mapSceneToView(
                event._scenePos
            )
            self.update_neural_predictions_vtick(mouse_point.x())

    def update_neural_predictions_vtick(self, x_pos=None):
        # Update the vertical line indicating the current frame in the neural predictions plot by setting the x position (x_pos) of the line
        if not self.neural_predictions_loaded:
            return
        if x_pos is not None:
            self.neural_predictions_vtick.setPos(x_pos)
            frame = int(x_pos)
        else:
            self.neural_predictions_vtick.setPos(self.parent.cframe)
            frame = self.parent.cframe
        # Check if x position is within the neural activity plot's current range of view
        if (
            not self.neural_predictions_plot.getViewBox().viewRange()[0][0]
            <= frame
            <= self.neural_predictions_plot.getViewBox().viewRange()[0][1]
        ):
            self.neural_predictions_plot.getViewBox().setXRange(frame, frame, padding=0)
            self.neural_predictions_plot.getViewBox().updateAutoRange()
        self.parent.current_frame_lineedit.setText(str(frame))

    def update_neural_data_vtick(self, x_pos=None):
        # Update the vertical line indicating the current frame in the neural data plot by setting the x position (x_pos) of the line
        if not self.neural_data_loaded:
            return
        if x_pos is not None:
            self.neural_activity_vtick.setPos(x_pos)
            frame = int(x_pos)
        else:
            self.neural_activity_vtick.setPos(self.parent.cframe)
            frame = self.parent.cframe
        # Check if x position is within the neural activity plot's current range of view
        if (
            not self.neural_activity_plot.getViewBox().viewRange()[0][0]
            <= frame
            <= self.neural_activity_plot.getViewBox().viewRange()[0][1]
        ):
            self.neural_activity_plot.getViewBox().setXRange(frame, frame, padding=0)
            self.neural_activity_plot.getViewBox().updateAutoRange()
        self.parent.current_frame_lineedit.setText(str(frame))