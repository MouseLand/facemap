"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
from PyQt5.QtWidgets import *
from PyQt5 import QtCore                                  
from PyQt5.QtCore import Qt
from ..tabs.videoplayer import VideoPlayer
from ..tabs.split_video_window import SplitVideoWindow
from facemap import utils
from facemap.tongue.segmentation_model import FMnet
import torch, os
from tqdm import tqdm
from facemap.tongue import segmentation_utils
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import pyqtgraph as pg
from ..sam2.popup_window import Sam2Popup

class SegmentationTab(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the UI
        self.setup_ui()

        # Initialize variables
        self.video_filenames = []
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model_path = None

    def setup_ui(self):
        # Set up the layout
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Set up the self.splitter
        self.splitter = QSplitter(Qt.Horizontal)
        # Set up the left panel with buttons
        button_panel = QWidget()
        button_layout = QVBoxLayout(button_panel)
        button_layout.setContentsMargins(0, 0, 0, 0)

        button_style = """
            QPushButton {
                background-color: rgb(237, 159, 114);
                color: black;
                border-radius: 5px;
                border: none;
                padding: 8px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: rgb(180, 108, 57);
            }
            QPushButton:pressed {
                background-color: rgb(196, 108, 57);
            }
        """

        # Add buttons with orange color
        video_button_groupbox = QGroupBox()
        video_button_groupbox.setLayout(QHBoxLayout())

        load_video_button = QPushButton("Load video")
        load_video_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white; font-size: 20px;")
        load_video_button.clicked.connect(self.add_video)
        load_video_button.setStyleSheet(button_style)
        video_button_groupbox.layout().addWidget(load_video_button)

        self.split_video_button = QPushButton("Split video")
        self.split_video_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white; font-size: 20px;")
        self.split_video_button.clicked.connect(self.split_video)
        self.split_video_button.setStyleSheet(button_style)
        video_button_groupbox.layout().addWidget(self.split_video_button)
        button_layout.addWidget(video_button_groupbox)

        model_option_groupbox = QGroupBox()
        model_option_groupbox.setLayout(QHBoxLayout())
        model_name_label = QLabel("Model:")
        model_name_label.setStyleSheet("color: lightgrey;")
        model_option_groupbox.layout().addWidget(model_name_label)

        # create dropdown menu for model selection
        self.model_dropdown = QComboBox()
        self.model_dropdown.setStyleSheet("background-color: rgb(196, 108, 57); color: white; font-size: 20px;")
        self.model_dropdown.addItem("SAM2")
        self.model_dropdown.addItem("FaceampNet")
        model_option_groupbox.layout().addWidget(self.model_dropdown)
        button_layout.addWidget(model_option_groupbox)

        run_segmentation_button = QPushButton("Run")
        run_segmentation_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white; font-size: 26px;")
        run_segmentation_button.clicked.connect(self.run_segmentation)
        run_segmentation_button.setStyleSheet(button_style)
        button_layout.addWidget(run_segmentation_button)

        path_button_groupbox = QGroupBox()
        path_button_groupbox.setLayout(QHBoxLayout())

        save_path_button = QPushButton("Set save path")
        save_path_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white; font-size: 20px;")
        save_path_button.clicked.connect(self.set_save_path)
        path_button_groupbox.layout().addWidget(save_path_button)

        load_model_button = QPushButton("Load model")
        load_model_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white; font-size: 20px;")
        load_model_button.clicked.connect(self.set_model_path)
        path_button_groupbox.layout().addWidget(load_model_button)

        button_layout.addWidget(path_button_groupbox)
        # Add a checkbox to save masks animation
        self.save_masks_checkbox = QCheckBox("Save masks animation")
        self.save_masks_checkbox.setStyleSheet("color: white;")
        button_layout.addWidget(self.save_masks_checkbox)

        # Add radio buttons for video views        
        video_view_groupbox = QGroupBox()
        # change style sheet to remove border
        video_view_groupbox.setStyleSheet("border: none;")
        video_view_groupbox.setLayout(QHBoxLayout())

        video_view_label = QLabel("Video View:")
        video_view_label.setAlignment(Qt.AlignCenter)
        video_view_label.setStyleSheet("color: lightgrey;")
        video_view_groupbox.layout().addWidget(video_view_label)

        self.video_view_group = QButtonGroup(button_panel)

        bottom_view_button = QRadioButton("Bottom")
        bottom_view_button.setStyleSheet("color: white;")
        bottom_view_button.setChecked(True)
        self.video_view_group.addButton(bottom_view_button)
        video_view_groupbox.layout().addWidget(bottom_view_button)

        side_view_button = QRadioButton("Side")
        side_view_button.setStyleSheet("color: white;")
        self.video_view_group.addButton(side_view_button)
        video_view_groupbox.layout().addWidget(side_view_button)

        button_layout.addWidget(video_view_groupbox)

        # Add a play button
        self.video_playback_groupbox = QGroupBox()
        self.video_playback_groupbox.setLayout(QHBoxLayout())

        self.play_button = QPushButton()
        self.video_playback_groupbox.layout().addWidget(self.play_button)

        self.frame_label = QLabel("0")
        self.frame_label.setStyleSheet("color: white;")
        self.video_playback_groupbox.layout().addWidget(self.frame_label)
        button_layout.addWidget(self.video_playback_groupbox)

        self.video_player = VideoPlayer(self.play_button)
        self.video_player2 = VideoPlayer(self.play_button)

        # Add the panels to the self.splitter
        self.splitter.addWidget(button_panel)
        self.splitter.addWidget(self.video_player)
        self.splitter.setStretchFactor(1, 3)

        # Set the style sheet for the dark theme and use white text
        dark_stylesheet = """
            QWidget {
                background-color: rgb(0,0,0);
            }
            QSplitter::handle {
                background-color: rgb(80, 80, 80);
            }
        """
        self.setStyleSheet(dark_stylesheet)
        self.light_stylesheet = """
            QWidget {
                background-color: rgb(255, 255, 255);
            }
        """
        # Add the self.splitter to the layout
        self.layout.addWidget(self.splitter, 0, 0)
        self.cumframes, self.Ly, self.Lx, self.containers = [], [], [], None

    def add_video(self):
        # Show file dialog to select video files
        file_dialog = QFileDialog(self)
        file_dialog.setStyleSheet(self.light_stylesheet)
        file_dialog.setNameFilter("Video Files (*.mj2 *.mp4 *.mkv *.avi *.mpeg *.mpg *.asf *m4v)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            # Add the selected files to the list of video filenames
            self.video_filenames += file_dialog.selectedFiles()
            self.cumframes, self.Ly, self.Lx, self.containers = utils.get_frame_details([[self.video_filenames[-1]]])
            self.video_player.load_video(self.cumframes, self.Ly, self.Lx, self.containers)#.abrir(self.video_filenames[-1])
            print(self.cumframes)

    def split_video(self):
        if self.video_filenames == []:
            # show qmessagebox to select video first
            msg = QMessageBox()
            msg.setStyleSheet(self.light_stylesheet)
            msg.setIcon(QMessageBox.Warning)
            msg.setText("Please load a video first")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok)
            msg.exec_()
        else:
            self.split_video_window = SplitVideoWindow(self, self.video_player)
            self.split_video_window.show()

    def add_video2(self, split_vline):
        self.video_player.load_video(self.cumframes, self.Ly, [split_vline], self.containers, crop=[0, self.Ly[0], 0, split_vline])
        self.video_player2.load_video(self.cumframes, self.Ly, [self.Lx[0]-split_vline], self.containers, crop=[0, self.Ly[0], split_vline, self.Lx[0]])
        self.split_video_button.setEnabled(False)
        self.splitter.addWidget(self.video_player2)

    def set_save_path(self):
        # Show file dialog to select save path
        file_dialog = QFileDialog(self)
        file_dialog.setStyleSheet(self.light_stylesheet)
        file_dialog.setFileMode(QFileDialog.Directory)
        if file_dialog.exec_():
            # Set the save path
            self.save_path = file_dialog.selectedFiles()[0]
            print("Save path set:", self.save_path)

    def set_model_path(self):
        # Show file dialog to select model file (*.pth)
        file_dialog = QFileDialog(self)
        file_dialog.setStyleSheet(self.light_stylesheet)
        file_dialog.setNameFilter("Model Files (*.pth)")
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            # Set the model path
            self.model_path = file_dialog.selectedFiles()[0]
            print("Model path set:", self.model_path)

    def run_segmentation(self):

        if self.model_dropdown.currentText() == "FaceampNet":
            self.run_faceampnet_segmentation()
        elif self.model_dropdown.currentText() == "SAM2":
            self.run_sam2_segmentation()

    def run_sam2_segmentation(self):
        print("Running SAM2 segmentation")
        popup = Sam2Popup(self, self.cumframes, self.Ly, self.Lx, self.containers)
        popup.exec_()

    def run_faceampnet_segmentation(self):
        # Get the video view
        video_view = self.video_view_group.checkedButton().text()
        # Run the segmentation
        segmentation_results = self.get_segmentation_results(video_view)
        # Show the results
        masks, edges = segmentation_results
        # save masks
        self.video_player.display_segmentation(masks, edges)
        self.save_segmentation_results(segmentation_results)
        print("Segmentation completed")

    def get_segmentation_results(self, video_view):
        model = FMnet() 
        model = model.to(self.device);
        if self.model_path is None:
            if video_view == "Bottom":
                model.load_state_dict(torch.load('/home/asyeda/Desktop/Hopkins/JHU_courses/DLCV/DLCV_final_project/fmnet_model/model_best.pth'))
                print("Bottom model weights loaded:", '/home/asyeda/Desktop/Hopkins/JHU_courses/DLCV/DLCV_final_project/fmnet_model/model_best.pth')
            elif video_view == "Side":
                model.load_state_dict(torch.load('/home/asyeda/Desktop/Hopkins/Jupyter_notebooks/segmentation_refinement/model_best.pth'))
                print("Side model weights loaded:", '/home/asyeda/Desktop/Hopkins/Jupyter_notebooks/segmentation_refinement/model_best.pth')
        else:
            model.load_state_dict(torch.load(self.model_path))
            print("Model weights loaded:", self.model_path)

        frames = segmentation_utils.get_img_from_video(self.video_filenames[-1])
        frames = segmentation_utils.preprocess_imgs(frames, resize_shape=(256, 256))

        print("Predicting masks")
        pred_masks, pred_edges = [], []
        for frame in tqdm(frames):
            frame = frame.unsqueeze(0).to(self.device)
            pred_mask, pred_edge, _ = predict(model, frame, self.device, sigmoid=True, threshold=0.5)
            pred_masks.append(pred_mask)
            pred_edges.append(pred_edge)

        return pred_masks, pred_edges
            

    def save_segmentation_results(self, segmentation_results):
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Plot restuls ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        if self.save_masks_checkbox.isChecked():
            print("Saving animation...")
            # Create an animation of video and model predictions
            fig, ax = plt.subplots(1, 3, figsize=(8, 5), dpi=300)

            start_idx = 0
            masks, edges = segmentation_results
            imgs = segmentation_utils.get_img_from_video(self.video_filenames[-1])  

            # Plot the original image
            img_plot = ax[0].imshow(imgs[start_idx].squeeze(), cmap='gray')
            ax[0].axis("off")
            ax[0].set_title("Frame: " + str(start_idx))
            mask_plot = ax[1].imshow(masks[start_idx].squeeze(), cmap='Greens', alpha=1, vmin=0, vmax=1)
            ax[1].axis("off")
            ax[1].set_title("Mask: " + str(start_idx))
            mask_edge_plot = ax[2].imshow(edges[start_idx].squeeze(), cmap='Reds', alpha=.4, vmin=0, vmax=1)
            ax[2].axis("off")
            ax[2].set_title("Edges: " + str(start_idx))

            def animate(i):
                img_plot.set_data(imgs[i].squeeze())
                ax[0].set_title("Frame: " + str(i))
                mask_plot.set_data(masks[i].squeeze())
                ax[1].set_title("Mask: " + str(i))
                mask_edge_plot.set_data(edges[i].squeeze())
                ax[2].set_title("Edges: " + str(i))
                return (img_plot, mask_plot, mask_edge_plot)

            anim = animation.FuncAnimation(fig, animate, frames=self.cumframes[-1]-5, interval=100, repeat=False, blit=True)
            # HTML(anim.to_html5_video())
            # save to mp4 using ffmpeg writer
            writervideo = animation.FFMpegWriter(fps=60, metadata=dict(artist='Your Name'), extra_args=['-vcodec', 'libx264'])
            filename = self.video_filenames[-1].split('/')[-1].split('.')[0]
            anim.save(os.path.join(self.save_path, filename+'_masks.mp4'), writer=writervideo)
            plt.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Save masks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print("Saving masks...")
        masks, edges = segmentation_results
        np.save(os.path.join(self.save_path, 'masks.npy'), masks)


def predict(net, im_input, device, sigmoid=True, threshold=0):
    # Predict
    net.eval()
    with torch.no_grad():
        mask_pred, mask_edges_pred, mask_dist_pred = net(im_input.to(device, dtype=torch.float32))

        if sigmoid:
            mask_pred = torch.sigmoid(mask_pred)
            mask_edges_pred = torch.sigmoid(mask_edges_pred)
        if threshold > 0:
            mask_pred[mask_pred > threshold] = 1
            mask_edges_pred[mask_edges_pred > threshold] = 1
            mask_pred[mask_pred <= threshold] = 0
            mask_edges_pred[mask_edges_pred <= threshold] = 0

    return mask_pred.cpu().numpy(), mask_edges_pred.cpu().numpy(), mask_dist_pred.cpu().numpy()
