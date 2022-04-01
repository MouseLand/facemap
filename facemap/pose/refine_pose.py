"""
Keypoints correction feature for new mouse videos
"""
import os
from natsort import natsorted
import pyqtgraph as pg
import numpy as np
from .. import utils
from PyQt5 import QtCore
import pandas as pd
from matplotlib import cm
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (QDialog, QWidget, QGridLayout, QLabel, QDialogButtonBox,
                            QSpinBox, QPushButton, QVBoxLayout, QRadioButton,
                            QHBoxLayout, QVBoxLayout, QButtonGroup, QGroupBox,
                            QListWidget, QAbstractItemView, QDesktopWidget)

# TO-DO:
# Fix add keypoint feature using right click -> doesn't always work
# Add frame number in popup window
# Edit functionality to delete keypoints using modifiers (ctrl, shift)

class KeypointsRefinementPopup(QDialog):
    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        
        self.setWindowTitle('Keypoints refinement')
        self.setStyleSheet("QDialog {background: 'black';}")

        self.verticalLayout = QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(-1, -1, -1, 0)
        # Set window size that is adjusted to the size of the window
        self.window_max_size = QDesktopWidget().screenGeometry(-1)

        self.show_step1_window()
        self.setLayout(self.verticalLayout)

        self.show()

    def update_window_size(self, frac=0.5):
        # Set the size of the window to be a fraction of the screen size
        self.resize(int(np.floor(self.window_max_size.width()*frac)), int(np.floor(self.window_max_size.height()*frac)))

    def clear_window(self):
        # Clear the window
        for i in reversed(range(self.verticalLayout.count())): 
            print(i, self.verticalLayout.itemAt(i).widget().objectName())
            self.verticalLayout.itemAt(i).widget().setParent(None)

    def update_window_title(self, title=None):
        if title is None:
            self.setWindowTitle('Keypoints refinement: frame {}/{}'.format(self.current_frame, self.spinBox_nframes.value()))
        else:
            self.setWindowTitle(title)

    def show_step1_window(self):
        # Open a dialog box to ask user if they want to refine keypoints from the current video or load previously refined keypoints
        # Use radio buttons to select between the two options
        self.update_window_title("Step 1: Select refinement option")
        self.update_window_size(0.3)

        # Create a group box to hold the radio buttons
        self.refine_keypoints_groupbox = QGroupBox()
        self.refine_keypoints_groupbox.setLayout(QHBoxLayout())

        # Create radio buttons
        self.refine_keypoints_radio_button_1 = QRadioButton("Refine keypoints for current video")
        self.refine_keypoints_radio_button_1.setChecked(True)
        self.refine_keypoints_radio_button_2 = QRadioButton("Load refined keypoints")
        self.refine_keypoints_radio_button_1.setStyleSheet("QRadioButton { font-size: 12pt; color: white; }")
        self.refine_keypoints_radio_button_2.setStyleSheet("QRadioButton { font-size: 12pt; color: white; }")

        # Add radio buttons to the group box
        self.refine_keypoints_groupbox.layout().addWidget(self.refine_keypoints_radio_button_1)
        self.refine_keypoints_groupbox.layout().addWidget(self.refine_keypoints_radio_button_2)

        # Add group box to the dialog box
        self.verticalLayout.addWidget(self.refine_keypoints_groupbox)

        # Create a button box to hold the buttons
        self.refine_keypoints_button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.refine_keypoints_button_box.accepted.connect(self.step1_done)
        self.refine_keypoints_button_box.rejected.connect(self.reject)
        # Add spacing between the buttons and padding around the buttons
        self.refine_keypoints_button_box.layout().setSpacing(10)
        self.refine_keypoints_button_box.layout().setContentsMargins(10, 10, 10, 10)
        
        self.verticalLayout.addWidget(self.refine_keypoints_button_box)

    def step1_done(self):
        self.clear_window()
        self.show_step2_window()

    def setup_pose_data(self):
        self.pose_data = pd.read_hdf(self.gui.poseFilepath[0], 'df_with_missing')
        self.bodyparts = np.array(pd.unique(self.pose_data.columns.get_level_values("bodyparts")))
        self.brushes, self.colors = self.get_brushes(self.bodyparts)

    def show_step2_window(self):
        """
        Add options for user to select from random frames drawn randomly from video or use KMeans to extract frames
        """
        self.get_step1_selection()
        if self.step1_selection == "refine":
            print("Refining keypoints")
            self.update_window_title("Step 2: Select frame extraction method")
            self.update_window_size(0.15)

            # Add a qlabel describing the purpose of the keypoints correction
            #self.label_horizontalLayout = QHBoxLayout()
            self.label = QLabel(self)
            self.label.setLayout(QHBoxLayout())
            self.label.setText("Please select a method for extracting frames and the number of frames for refinement.\nExtracted frames from the video will be used for retraining the model on the refined keypoints. Recommended #frames for refinement are 20-25.")
            self.label.setStyleSheet("QLabel { font-size: 12pt; color: grey; }")
            self.label.setWordWrap(True)
            #self.label_horizontalLayout.addWidget(self.label)
            self.verticalLayout.addWidget(self.label) #addLayout(self.label_horizontalLayout)

            # Add a group box to hold the radio buttons
            self.frames_selection_mode_group = QGroupBox()
            self.frames_selection_mode_group.setLayout(QHBoxLayout())
            self.frames_selection_mode_group.setStyleSheet("QGroupBox { font-size: 12pt; color: white; border : 0px solid black; }")
            # Create radio buttons
            self.frames_selection_mode_radio_button_1 = QRadioButton("Random frames")
            self.frames_selection_mode_radio_button_1.setChecked(True)
            self.frames_selection_mode_radio_button_2 = QRadioButton("K-means")
            self.frames_selection_mode_radio_button_1.setStyleSheet("QRadioButton { font-size: 12pt; color: white; }")
            self.frames_selection_mode_radio_button_2.setStyleSheet("QRadioButton { font-size: 12pt; color: white; }")
            # Add radio buttons to the group box
            self.frames_selection_mode_group.layout().addWidget(self.frames_selection_mode_radio_button_1)
            self.frames_selection_mode_group.layout().addWidget(self.frames_selection_mode_radio_button_2)
            # Add group box to the dialog box
            self.verticalLayout.addWidget(self.frames_selection_mode_group)

            # Create a horizontal layout for the spinboxes
            self.frame_number_selection_box =  QGroupBox()
            self.frame_number_selection_box.setLayout(QHBoxLayout())
            # Add a QLabel and QSpinBox to the layout for the number of frames to be processed
            self.label_nframes = QLabel('# Frames for refinement:')
            self.label_nframes.setStyleSheet("QLabel { font-size: 12pt; color: white; }")
            self.spinBox_nframes = QSpinBox()
            self.spinBox_nframes.setRange(1, self.gui.nframes)
            self.spinBox_nframes.setValue(25)
            self.spinBox_nframes.setFixedWidth(100)
            # Add a QLabel and QSpinBox to the horizontal layout
            self.frame_number_selection_box.layout().addWidget(self.label_nframes)
            self.frame_number_selection_box.layout().addWidget(self.spinBox_nframes)
            self.verticalLayout.addWidget(self.frame_number_selection_box) #addLayout(self.horizontalLayout)

            # Create a button box to hold the buttons
            self.frames_selection_mode_button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
            self.frames_selection_mode_button_box.accepted.connect(self.step2_done)
            self.frames_selection_mode_button_box.rejected.connect(self.reject)
            # Add buttons to the button box
            self.verticalLayout.addWidget(self.frames_selection_mode_button_box)

        elif self.step1_selection == "load":
            print("Loading keypoints")
            self.update_window_title("Step 2: Select keypoints to load")
            self.update_window_size(0.3)
            LC = PoseFileListChooser('Choose files', self.gui)
            result = LC.exec_()

        else:
            print("No selection made")
            return
        pass

    def step2_done(self):
        self.clear_window()
        print("Step 2 done")
        mode = self.get_step2_frame_selection_mode()
        print(mode)
        #self.show_step3_window()

    def get_step1_selection(self):
        # Get the selected option from the radio buttons
        if self.refine_keypoints_radio_button_1.isChecked():
            self.step1_selection = 'refine'
        elif self.refine_keypoints_radio_button_2.isChecked():
            self.step1_selection = 'load'
        else:
            self.step1_selection = None

    def get_step2_frame_selection_mode(self):
        if self.frames_selection_mode_radio_button_1.isChecked():
            return "random"
        elif self.frames_selection_mode_radio_button_2.isChecked():
            return "kmeans"
        else:
            return None

    def tbd(self):
        pass
    """
        if self.poseFileLoaded and self.pose_model is not None:
            apply_keypoints_correction(self)
        elif self.poseFileLoaded:
            self.setup_pose_model()
            apply_keypoints_correction(self)
        else:
            # Create a message box to ask user to process the keypoints or load a pose file
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Please load a pose file first or process the keypoints")
            msg.setWindowTitle("No pose file loaded")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
    """

    def show_step3_window(self):
        self.overall_horizontalLayout = QHBoxLayout(self)

        self.frame_horizontalLayout = QHBoxLayout()
        self.win = pg.GraphicsLayoutWidget()
        self.win.setObjectName("Keypoints refinement")
        self.frame_win = self.win.addViewBox(invertY=True)
        self.keypoints_scatterplot = KeypointsGraph(parent=self) 
        self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.frame_horizontalLayout.addWidget(self.win)
        
        self.current_frame = 0

        # Create a horizontal layout for the buttons
        self.horizontalLayout_2 = QHBoxLayout()
        self.horizontalLayout_2.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_2.setObjectName("horizontalLayout")
        self.cancel_button = QPushButton('Cancel')
        self.cancel_button.clicked.connect(self.close)
        self.horizontalLayout_2.addWidget(self.cancel_button)
        # Add a button for next step
        self.ok_button = QPushButton('Ok')
        self.ok_button.setDefault(True)
        self.ok_button.clicked.connect(self.display_frames_w_keypoints)
        self.horizontalLayout_2.addWidget(self.ok_button)
        
        # Position buttons
        self.widget = QWidget(self)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.verticalLayout.addWidget(self.widget)

        # Define buttons for main window
        self.previous_button = QPushButton('Previous')
        self.previous_button.clicked.connect(self.previous_frame)
        self.previous_button.hide()
        # Add a button for next step
        self.next_button = QPushButton('Next')
        self.next_button.setDefault(True)
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.hide()
        self.finish_button = QPushButton('Finish')
        self.finish_button.clicked.connect(self.retrain_model)
        self.finish_button.hide()
        self.horizontalLayout_2.addWidget(self.previous_button)
        self.horizontalLayout_2.addWidget(self.next_button)

        self.finish_horozontalLayout = QHBoxLayout()
        self.finish_horozontalLayout.addWidget(self.finish_button)
        self.verticalLayout.addLayout(self.finish_horozontalLayout)

        # Radio buttons group for selecting the bodyparts to be corrected
        self.radio_verticalLayout = QVBoxLayout()
        # Add a label for the radio buttons
        self.radio_label = QLabel('Bodyparts')
        self.radio_label.hide()
        self.radio_verticalLayout.addWidget(self.radio_label)
        self.radio_buttons_group = QButtonGroup()
        self.radio_buttons_group.setExclusive(True)
        self.radio_buttons_group.setObjectName("radio_buttons_group")
        self.radio_buttons = []
        for i, bodypart in enumerate(self.bodyparts):
            self.radio_buttons.append(QRadioButton(bodypart))
            self.radio_buttons[i].hide()
            # Change color of radio button
            color  = QColor(self.colors[i][0], self.colors[i][1], self.colors[i][2])
            alpha  = 150
            values = "{r}, {g}, {b}, {a}".format(r = color.red(),
                                                g = color.green(),
                                                b = color.blue(),
                                                a = alpha
                                                )
            self.radio_buttons[i].setStyleSheet("QRadioButton { background-color: rgba("+values+"); border: 1px solid black; }")
            self.radio_buttons[i].setObjectName(bodypart)
            self.radio_buttons[i].clicked.connect(self.radio_button_clicked)
            self.radio_buttons_group.addButton(self.radio_buttons[i])
            self.radio_verticalLayout.addWidget(self.radio_buttons[i])

        self.overall_horizontalLayout.addLayout(self.verticalLayout)
        self.overall_horizontalLayout.addLayout(self.radio_verticalLayout)

    # Add a keyPressEvent for deleting the selected keypoint using the delete key and set the value to NaN in the dataframe
    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.delete_keypoint()
        else:
            return

    def radio_button_clicked(self):
        # Change background color of the selected radio button to None
        for i, button in enumerate(self.radio_buttons):
            if button.isChecked():
                color  = QColor(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                alpha  = 0
                values = "{r}, {g}, {b}, {a}".format(r = color.red(),
                                                    g = color.green(),
                                                    b = color.blue(),
                                                    a = alpha
                                                    )
                button.setStyleSheet("QRadioButton { background-color: rgba("+values+"); border: 1px solid black; }")
            else:
                color  = QColor(self.colors[i][0], self.colors[i][1], self.colors[i][2])
                alpha  = 150
                values = "{r}, {g}, {b}, {a}".format(r = color.red(),
                                                    g = color.green(),
                                                    b = color.blue(),
                                                    a = alpha
                                                    )
                button.setStyleSheet("QRadioButton { background-color: rgba("+values+"); border: 0px solid black; }")

    def delete_keypoint(self):
        # Get index of radio button that is selected
        index = self.radio_buttons_group.checkedId()
        # Get the bodypart that is selected
        bodypart = self.radio_buttons_group.button(index).text()
        selected_items = np.where(self.bodyparts == bodypart)[0]        
        # Delete the selected keypoint
        #selected_items = self.keypoints_scatterplot.hover()
        for i in selected_items:
            if not np.isnan(self.keypoints_scatterplot.data['pos'][i]).any():
                self.keypoints_scatterplot.data['pos'][i] = np.nan
                # Update radio buttons
                if i < len(self.radio_buttons)-1:
                    self.radio_buttons[i+1].setChecked(True)
                    self.radio_buttons[i+1].clicked.emit(True)
                self.keypoints_scatterplot.updateGraph(dragged=True)
            else:
                return

    def clear_window_old(self):
        # Hide frame_win
        self.frame_horizontalLayout.removeWidget(self.win)
        self.win.setParent(None)

        # Clear the popup and display the frames with keypoints
        self.verticalLayout.removeWidget(self.label)
        self.label.setParent(None)
        self.verticalLayout.removeWidget(self.label_nframes)
        self.label_nframes.setParent(None)
        self.verticalLayout.removeWidget(self.spinBox_nframes)
        self.spinBox_nframes.setParent(None)
        self.horizontalLayout_2.removeWidget(self.cancel_button)
        self.cancel_button.setParent(None)
        self.horizontalLayout_2.removeWidget(self.ok_button)
        self.ok_button.setParent(None)
        self.previous_button.hide()
        self.next_button.hide()
        self.finish_button.hide()
        for i, bodypart in enumerate(self.bodyparts):
            self.radio_buttons[i].hide()
            self.radio_label.hide()

    def show_main_features(self):
        self.clear_window()
        # Show main page features
        self.label_horizontalLayout.addWidget(self.label)
        self.label.setParent(self)
        self.verticalLayout.addLayout(self.label_horizontalLayout)

        # Add a QLabel and QSpinBox to the horizontal layout
        self.horizontalLayout.addWidget(self.label_nframes)
        self.label_nframes.setParent(self)
        self.horizontalLayout.addWidget(self.spinBox_nframes)
        self.spinBox_nframes.setParent(self)
        self.verticalLayout.addLayout(self.horizontalLayout)

        # Add buttons to the horizontal layout
        self.horizontalLayout_2.addWidget(self.cancel_button)
        self.cancel_button.setParent(self)
        self.horizontalLayout_2.addWidget(self.ok_button)
        self.ok_button.setParent(self)
        self.verticalLayout.addLayout(self.horizontalLayout_2)

        for i, _ in enumerate(self.bodyparts):
            self.radio_buttons[i].hide()
            self.radio_label.hide()

        # Reset frame counter
        self.current_frame = 0
        self.update_window_title("Keypoints refinement")

    def display_frames_w_keypoints(self):
        self.clear_window()
        self.frame_horizontalLayout.addWidget(self.win)
        self.win.setParent(self)
        self.verticalLayout.addLayout(self.frame_horizontalLayout)

        # Select frames for keypoints correction
        self.random_frames_ind = sorted(np.random.choice(self.gui.nframes, self.spinBox_nframes.value(), replace=False)) 
        for i, bodypart in enumerate(self.bodyparts):
            self.radio_buttons[i].show()
            self.radio_label.show()
        self.previous_button.show()
        self.next_button.show()
        self.finish_button.show()
        self.next_button.setDefault(True)
        self.next_frame()
        
    def previous_frame(self):
        # Go to previous frame
        self.current_frame -= 1
        # Update the current frame
        if self.current_frame <= self.spinBox_nframes.value() and self.current_frame > 0:
            self.frame_win.clear()
            self.next_button.show()
            self.next_button.setEnabled(True)
            selected_frame = utils.get_frame(self.random_frames_ind[self.current_frame], self.gui.nframes, 
                                            self.gui.cumframes, self.gui.video)[0] 
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.plot_keypoints(self.random_frames_ind[self.current_frame])
            self.update_window_title()
        else:
            self.show_main_features()
            
    def next_frame(self):
        # Display the next frame in list of random frames with keypoints
        self.previous_button.show()
        self.current_frame += 1
        if self.current_frame < self.spinBox_nframes.value():
            self.frame_win.clear()
            selected_frame = utils.get_frame(self.random_frames_ind[self.current_frame], self.gui.nframes, 
                                            self.gui.cumframes, self.gui.video)[0] 
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.update_window_title()
            self.plot_keypoints(self.random_frames_ind[self.current_frame])
            self.next_button.setText('Next')
        else:
            self.next_button.setEnabled(False)
        self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.win.show()

    def retrain_model(self):
        # Retrain the model on the selected frames
        self.gui.retrain_model(self.random_frames_ind)
        self.close()

    def plot_keypoints(self, frame_ind):
        # Plot the keypoints of the selected frames
        plot_pose_data = self.pose_data.iloc[frame_ind]
        # Append pose data to list for each video_id
        x = plot_pose_data[::3].values
        y = plot_pose_data[1::3].values
        # Add a scatter plot item to the window for each bodypart
        self.keypoints_scatterplot.setData(pos=np.array([x, y]).T, symbolBrush=self.brushes, symbolPen=self.colors,
                                                symbol='o',  brush=self.brushes, hoverable=True, hoverSize=25, 
                                                hoverSymbol="x", pxMode=True, hoverBrush='r', name=self.bodyparts, 
                                                data=self.bodyparts, size=20) 
        self.frame_win.addItem(self.keypoints_scatterplot)                                

    def get_brushes(self, bodyparts):
        num_classes = len(bodyparts)
        colors = cm.get_cmap('jet')(np.linspace(0, 1., num_classes))
        colors *= 255
        colors = colors.astype(int)
        colors[:,-1] = 230
        brushes = [pg.mkBrush(color=c) for c in colors]
        return brushes, colors

def apply_keypoints_correction(guiObject):
    """
    Apply keypoints correction to the predicted pose data for re-training the model
    """
    # Select poseFilepath for keypoints correction
    LC = PoseFileListChooser('Choose files', guiObject)
    result = LC.exec_()
    KeypointsRefinementPopup(guiObject)

def save_pose_data(gui, pose_data, frame_ind):
    # Save the refined keypoints data to a file
    filepath = gui.poseFilepath[0].split("_FacemapPose.h5")[0]
    pose_data = pose_data.loc[frame_ind]
    pose_data.to_hdf(filepath+'_FacemapPoseRefined.h5', "df_with_missing", mode="w")


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
        self.text = kwds.pop('text', [])
        self.data = kwds
        if 'pos' in self.data and len(kwds)==0:
            npts = self.data['pos'].shape[0]
            self.data['data'] = np.empty(npts, dtype=[('index', str)])
            self.data['data'] = kwds['name']
            self.data['data']['index'] = np.arange(npts)
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
        point_hovered = np.where(self.scatter.data['hovered'])[0]
        return point_hovered

    def updateGraph(self, dragged=False):
        pg.GraphItem.setData(self, **self.data)
        for i,item in enumerate(self.textItems):
            item.setPos(*self.data['pos'][i])
        if dragged:
            keypoints_refined = self.getData()
            self.update_pose_file(keypoints_refined)
        
    # Write a function that tracks all the keypoints positions for all frames and save them in a file (for later use)
    def update_pose_file(self, keypoints_refined):
        # Get values for the keypoints from the GUI and save them in a file
        x_coord = keypoints_refined[:,0]
        y_coord = keypoints_refined[:,1]
        frame_ind = self.parent.random_frames_ind[self.parent.current_frame]
        for i, bp in enumerate(self.parent.bodyparts):
            self.parent.pose_data.loc[frame_ind]['Facemap'][bp]['x'] = x_coord[i]
            self.parent.pose_data.loc[frame_ind]['Facemap'][bp]['y'] = y_coord[i]
        save_pose_data(self.parent.gui, self.parent.pose_data, self.parent.random_frames_ind)

    def getData(self):
        return self.data['pos']

    def mousePressEvent(self, QMouseEvent):
        if QMouseEvent.button() == QtCore.Qt.RightButton:
            self.right_click_keypoint(QMouseEvent)
        elif QMouseEvent.button() == QtCore.Qt.LeftButton:
            super().mousePressEvent(QMouseEvent)

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
                ind = pts[0].index() #pts[0].data()[0]
                # Create a bool array of length equal to the number of points in the scatter plot
                # and set it to False
                bool_arr = np.zeros(len(self.data['pos']), dtype=bool)
                # Set the value of the bool array to True for the point that was clicked
                bool_arr[ind] = True
                self.dragOffset = self.data['pos'][bool_arr] - pos
            elif ev.isFinish():
                self.dragPoint = None
                return
            else:
                if self.dragPoint is None:
                    ev.ignore()
                    return
            
            ind = self.dragPoint.index() #self.dragPoint.data()[0]
            # Create a bool array of length equal to the number of points in the scatter plot
            bool_arr = np.zeros(len(self.data['pos']), dtype=bool)
            bool_arr[ind] = True
            self.data['pos'][bool_arr] = ev.pos() + self.dragOffset[0]
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
        for i, bp in enumerate(self.parent.bodyparts):
            if self.parent.radio_buttons[i].isChecked():
                bp_selected = bp
                break
        # Get position of the selected bodypart
        selected_bp_ind = np.where(self.data['data'] == bp_selected)[0]
        # Check if position of bodypart is nan
        selected_bp_pos = self.data['pos'][selected_bp_ind][0]
        x, y = selected_bp_pos[0], selected_bp_pos[1]
        # If keypoint is deleted, then add it back using the user selected position
        if np.isnan(x) and np.isnan(y):
            # Get position of mouse from the mouse event
            add_point_pos = mouseEvent.pos()
            add_x, add_y = add_point_pos.x(), add_point_pos.y()
            keypoints_refined = self.getData()
            keypoints_refined[selected_bp_ind] = [add_x, add_y]
            # Add a keypoint to the scatter plot at the clicked position to add the bodypart
            self.scatter.setData(pos=np.array(keypoints_refined), symbolBrush=self.parent.brushes, 
                                                    symbolPen=self.parent.colors, symbol='o', brush=self.parent.brushes,
                                                    hoverable=True, hoverSize=25, hoverSymbol="x", pxMode=True, hoverBrush='r',
                                                    name=self.parent.bodyparts, data=self.parent.bodyparts, size=20) 
            # Update the data
            self.updateGraph()
            # Update the pose file   
            keypoints_refined = self.getData()
            self.update_pose_file(keypoints_refined)         
            # Check the next bodypart in the list of radio buttons
            if selected_bp_ind[0] < len(self.parent.bodyparts) - 1:
                self.parent.radio_buttons[selected_bp_ind[0]+1].setChecked(True)
                self.parent.radio_buttons[selected_bp_ind[0]+1].clicked.emit(True)
            else:
                self.parent.radio_buttons[0].setChecked(True)
                self.parent.radio_buttons[0].clicked.emit(True)
        else:
            return

class PoseFileListChooser(QDialog):
    def __init__(self, title, parent):
        super().__init__(parent)
        self.setGeometry(300,300,320,320)
        self.setWindowTitle(title)
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        layout.addWidget(QLabel('click to select videos (none selected => all used)'),0,0,1,1)
        self.list = QListWidget(parent)
        for f in natsorted(parent.poseFilepath):
            self.list.addItem(f)
        layout.addWidget(self.list,1,0,7,4)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        done = QPushButton('done')
        done.clicked.connect(lambda: self.exit_list(parent))
        layout.addWidget(done,8,0,1,1)

    def exit_list(self, parent):
        parent.keypoint_correction_file = []
        items = self.list.selectedItems()
        for i in range(len(items)):
            parent.keypoint_correction_file.append(str(self.list.selectedItems()[i].text()))
        self.accept()

