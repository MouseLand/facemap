"""
Keypoints correction feature for new mouse videos
"""
from natsort import natsorted
import pyqtgraph as pg
import numpy as np
from .. import utils
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (QDialog, QWidget, QGridLayout, QLabel,
                            QSpinBox, QPushButton, QVBoxLayout,
                            QHBoxLayout, QVBoxLayout, QCheckBox, 
                            QListWidget, QAbstractItemView)

def apply_keypoints_correction(guiObject):
    """
    Apply keypoints correction to the predicted pose data for re-training the model
    """
    # Select poseFilepath for keypoints correction
    LC = PoseFileListChooser('Choose files', guiObject)
    result = LC.exec_()
    print(LC, result,guiObject.keypoint_correction_file)
    KeypointsRefinementPopup(guiObject)

# Following used to check cropped sections of frames
class KeypointsRefinementPopup(QDialog):
    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui

        self.setWindowTitle('Keypoints refinement')
        self.verticalLayout = QVBoxLayout(self)
        
        self.frame_horizontalLayout = QHBoxLayout()
        self.win = pg.GraphicsLayoutWidget()
        self.win.setObjectName("Keypoints refinement")
        self.frame_win = self.win.addViewBox(invertY=True)
        self.frame_horizontalLayout.addWidget(self.win)
        self.current_frame = 0

        # Add a qlabel describing the purpose of the keypoints correction
        self.label_horizontalLayout = QHBoxLayout()
        self.label = QLabel(self)
        self.label.setText("Please set following options for keypoints correction. \n Random frames from the video will be selected for retraining the model on the corrected frames. \n Recommended #frames to use for retraining the model are 20-25.")
        self.label_horizontalLayout.addWidget(self.label)
        self.verticalLayout.addLayout(self.label_horizontalLayout)

        # Create a horizontal layout for the spinboxes
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setSpacing(0)
        # Add a QLabel and QSpinBox to the layout for the number of frames to be processed
        self.label_nframes = QLabel('# Frames for refinement:')
        self.spinBox_nframes = QSpinBox()
        self.spinBox_nframes.setRange(1, self.gui.nframes)
        self.spinBox_nframes.setValue(25)
        # Set width of spinbox
        self.spinBox_nframes.setFixedWidth(100)
        self.verticalLayout.addWidget(self.label_nframes)
        self.verticalLayout.addWidget(self.spinBox_nframes)
        # Add a QLabel and QSpinBox to the horizontal layout
        self.horizontalLayout.addWidget(self.label_nframes)
        self.horizontalLayout.addWidget(self.spinBox_nframes)
        self.verticalLayout.addLayout(self.horizontalLayout)

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
        self.horizontalLayout_2.addWidget(self.previous_button)
        self.horizontalLayout_2.addWidget(self.next_button)

        self.show()

    def clear_window(self):
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

    def display_frames_w_keypoints(self):
        self.clear_window()
        self.frame_horizontalLayout.addWidget(self.win)
        self.win.setParent(self)
        self.verticalLayout.addLayout(self.frame_horizontalLayout)

        # Select frames for keypoints correction
        self.random_frames_ind = np.random.choice(self.gui.nframes, self.spinBox_nframes.value(), replace=False)   
        self.previous_button.show()
        self.next_button.show()
        self.next_button.setDefault(True)
        self.next_frame()
        
    def previous_frame(self):
        # Go to previous frame
        self.win.setObjectName("Frame " + str(self.current_frame-1))
        # Update the current frame
        if self.current_frame < self.spinBox_nframes.value() and self.current_frame > 0:
            self.frame_win.clear()
            self.current_frame -= 1
            selected_frame = utils.get_frame(self.random_frames_ind[self.current_frame], self.gui.nframes, self.gui.cumframes, self.gui.video)[0] 
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
        else:
            self.show_main_features()
            
    def next_frame(self):
        # Display the next frame in list of random frames with keypoints
        self.win.setObjectName("Frame " + str(self.current_frame))
        # Update the current frame
        self.previous_button.show()
        if self.current_frame < self.spinBox_nframes.value():
            self.frame_win.clear()
            print(self.current_frame)
            selected_frame = utils.get_frame(self.random_frames_ind[self.current_frame], self.gui.nframes, self.gui.cumframes, self.gui.video)[0] 
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
            self.next_button.setText('Next')
        else:
            self.next_button.setText('Finish')
            self.next_button.clicked.connect(self.retrain_model)
        self.current_frame += 1
        self.frame_win.setAspectLocked(True, QtCore.Qt.IgnoreAspectRatio)
        self.frame_win.setMouseEnabled(False, False)
        self.frame_win.setMenuEnabled(False)
        self.win.show()

    def retrain_model(self):
        # Retrain the model on the selected frames
        #self.gui.retrain_model(self.random_frames_ind)
        self.close()

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

