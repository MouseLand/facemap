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
from PyQt5.QtWidgets import (QDialog, QWidget, QGridLayout, QLabel,
                            QSpinBox, QPushButton, QVBoxLayout, QRadioButton,
                            QHBoxLayout, QVBoxLayout, QButtonGroup, 
                            QListWidget, QAbstractItemView)

def apply_keypoints_correction(guiObject):
    """
    Apply keypoints correction to the predicted pose data for re-training the model
    """
    # Select poseFilepath for keypoints correction
    LC = PoseFileListChooser('Choose files', guiObject)
    result = LC.exec_()
    KeypointsRefinementPopup(guiObject)

# Following used to check cropped sections of frames
class KeypointsRefinementPopup(QDialog):
    def __init__(self, gui):
        super().__init__(gui)
        self.gui = gui
        self.pose_data = pd.read_hdf(self.gui.poseFilepath[0], 'df_with_missing')
        self.bodyparts = np.array(pd.unique(self.pose_data.columns.get_level_values("bodyparts")))

        self.setWindowTitle('Keypoints refinement')

        self.overall_horizontalLayout = QHBoxLayout(self)
        self.verticalLayout = QVBoxLayout(self)
        
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

        # Add a qlabel describing the purpose of the keypoints correction
        self.label_horizontalLayout = QHBoxLayout()
        self.label = QLabel(self)
        self.label.setText("Please set following options for keypoints correction. \n Random frames from the video will be selected for retraining the model on the corrected frames. \n Recommended #frames to use for retraining the model are 20-25. Please note that clicking OK will reset any refined labels and select a new set of random frames.")
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
        self.radio_verticalLayout.addWidget(self.radio_label)
        self.radio_buttons_group = QButtonGroup()
        self.radio_buttons_group.setExclusive(True)
        #self.radio_buttons_group.buttonClicked.connect(self.update_keypoints)
        self.radio_buttons_group.setObjectName("radio_buttons_group")
        self.radio_buttons = []
        for i, bodypart in enumerate(self.bodyparts):
            self.radio_buttons.append(QRadioButton(bodypart))
            self.radio_buttons[i].hide()
            self.radio_buttons[i].setObjectName(bodypart)
            self.radio_buttons_group.addButton(self.radio_buttons[i])
            self.radio_verticalLayout.addWidget(self.radio_buttons[i])

        self.overall_horizontalLayout.addLayout(self.verticalLayout)
        self.overall_horizontalLayout.addLayout(self.radio_verticalLayout)

        self.show()

    # Add a keyPressEvent for deleting the selected keypoint using the delete key and set the value to NaN in the dataframe
    def keyPressEvent(self, ev):
        if ev.key() in (QtCore.Qt.Key_Backspace, QtCore.Qt.Key_Delete):
            self.delete_keypoint()
        else:
            return

    def delete_keypoint(self):
        # Delete the selected keypoint
        selected_items = self.keypoints_scatterplot.hover()
        if selected_items is not None:
            for i in selected_items:
                self.keypoints_scatterplot.data['pos'][i] = np.nan
            self.keypoints_scatterplot.updateGraph(dragged=True)
        else:
            print("Please hover over a keypoint to delete it")

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

        for i, bodypart in enumerate(self.bodyparts):
            self.radio_buttons[i].hide()
            self.radio_label.hide()

        # Reset frame counter
        self.current_frame = 0

    def display_frames_w_keypoints(self):
        self.clear_window()
        self.frame_horizontalLayout.addWidget(self.win)
        self.win.setParent(self)
        self.verticalLayout.addLayout(self.frame_horizontalLayout)

        # Select frames for keypoints correction
        self.random_frames_ind = np.random.choice(self.gui.nframes, self.spinBox_nframes.value(), replace=False)   
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
        print("current frame: ", self.current_frame)
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
        else:
            self.show_main_features()
            
    def next_frame(self):
        # Display the next frame in list of random frames with keypoints
        self.previous_button.show()
        self.current_frame += 1
        print("current frame: ", self.current_frame)
        if self.current_frame < self.spinBox_nframes.value():
            self.frame_win.clear()
            selected_frame = utils.get_frame(self.random_frames_ind[self.current_frame], self.gui.nframes, 
                                            self.gui.cumframes, self.gui.video)[0] 
            self.img = pg.ImageItem(selected_frame)
            self.frame_win.addItem(self.img)
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
        #self.gui.retrain_model(self.random_frames_ind)
        self.close()

    def plot_keypoints(self, frame_ind):
        # Plot the keypoints of the selected frames
        plot_pose_data = self.pose_data.iloc[frame_ind]
        # Append pose data to list for each video_id
        x = plot_pose_data[::3].values
        y = plot_pose_data[1::3].values
        brushes, colors = self.get_brushes(self.bodyparts)
        # Add a scatter plot item to the window for each bodypart
        self.keypoints_scatterplot.setData(pos=np.array([x, y]).T, symbolBrush=brushes, symbolPen=colors,
                                                symbol='o',  brush=brushes, hoverable=True, hoverSize=25, 
                                                hoverSymbol="x", pxMode=True, hoverBrush='r', name=self.bodyparts, 
                                                data=self.bodyparts, size=20) 
        self.frame_win.addItem(self.keypoints_scatterplot)                                

    def get_brushes(self, bodyparts):
        num_classes = len(bodyparts)
        colors = cm.get_cmap('jet')(np.linspace(0, 1., num_classes))
        colors *= 255
        colors = colors.astype(int)
        colors[:,-1] = 200
        brushes = [pg.mkBrush(color=c) for c in colors]
        return brushes, colors

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

    def mouseDragEvent(self, ev):
        if ev.button() != QtCore.Qt.LeftButton:
            ev.ignore()
            return
        
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
        
    def keypoint_clicked(self, pts):
        return None
    
    # Add feature for adding a keypoint to the scatterplot

# TO-DO:
# Write a function that loads the keypoints from a file and uses them to re-train the model
# Add a feature for using the retrained model for the next pose prediction

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

def save_pose_data(gui, pose_data, frame_ind):
    # Save the refined keypoints data to a file
    filepath = gui.poseFilepath[0].split("_FacemapPose.h5")[0]
    pose_data = pose_data.loc[frame_ind]
    pose_data.to_hdf(filepath+'_FacemapPoseRefined.h5', "df_with_missing", mode="w")

