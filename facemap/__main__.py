"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import sys, os
import numpy as np
import argparse
from distutils.util import strtobool
from PyQt5.QtWidgets import QApplication, QDialog, QVBoxLayout, QPushButton, QDesktopWidget
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon

from facemap import process, version_str
from facemap.gui import gui
from facemap.tongue import tongue_gui


def main():
    ops = np.load("ops.npy")
    ops = ops.item()


class LaunchDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Facemap")
        self.setGeometry(200, 200, 450, 500)

        layout = QVBoxLayout()

        # Add a logo to the dialog
        logo_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "mouse.png"
        )
        logo = QIcon()
        logo.addFile(logo_path, QtCore.QSize(256, 256))
        logo_label = QPushButton("", self)
        logo_label.setIcon(logo)
        logo_label.setIconSize(QtCore.QSize(256, 256))
        logo_label.setFlat(True)
        layout.addWidget(logo_label)

        pose_estimation_button = QPushButton("Pose Estimation", self)
        pose_estimation_button.clicked.connect(self.on_pose_estimation_clicked)
        layout.addWidget(pose_estimation_button)

        tongue_gui_button = QPushButton("Tongue segmentation", self)
        tongue_gui_button.clicked.connect(self.on_tongue_gui_clicked)
        layout.addWidget(tongue_gui_button)

        self.setLayout(layout)
        self.selected_option = None

        self.center_on_screen()  # Center the dialog on the screen


    def center_on_screen(self):
        screen_geometry = QDesktopWidget().screenGeometry()
        dialog_geometry = self.geometry()
        x = (screen_geometry.width() - dialog_geometry.width()) // 2
        y = (screen_geometry.height() - dialog_geometry.height()) // 2
        self.move(x, y)

    def on_pose_estimation_clicked(self):
        self.selected_option = "Pose estimation"
        self.accept()

    def on_tongue_gui_clicked(self):
        self.selected_option = "Tongue segmentation"
        self.accept()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Movie files")
    parser.add_argument("--ops", default=[], type=str, help="options")
    parser.add_argument(
        "--movie", default=None, nargs="+", type=str, help="Absolute path to video(s)"
    )
    # Currently supports loading movie files recorded simultaneously
    parser.add_argument(
        "--keypoints",
        default=None,
        nargs="+",
        type=str,
        help="Absolute path to keypoints file (*.h5)",
    )
    parser.add_argument(
        "--proc_npy",
        default=None,
        type=str,
        help="Absolute path to proc file (*_proc.npy)",
    )
    parser.add_argument(
        "--neural_activity",
        default=None,
        type=str,
        help="Absolute path to neural activity file (*.npy)",
    )
    parser.add_argument(
        "--neural_prediction",
        default=None,
        type=str,
        help="Absolute path to neural prediction file (*.npy)",
    )
    parser.add_argument(
        "--tneural",
        default=None,
        type=str,
        help="Absolute path to neural timestamps file (*.npy)",
    )
    parser.add_argument(
        "--tbehavior",
        default=None,
        type=str,
        help="Absolute path to behavior timestamps file (*.npy)",
    )
    parser.add_argument("--savedir", default=None, type=str, help="save directory")
    # Add a flag to autoload keypoints in the same directory as the movie
    parser.add_argument(
        "--autoload_keypoints",
        dest="autoload_keypoints",
        type=lambda x: bool(strtobool(x)),
        help="Automatically load keypoints in the same directory as the movie",
    )
    parser.set_defaults(autoload_keypoints=True)

    # Add a flag to autoload proc in the same directory as the movie
    parser.add_argument(
        "--autoload_proc",
        dest="autoload_proc",
        type=lambda x: bool(strtobool(x)),
        help="Automatically load *_proc.npy in the same directory as the movie",
    )
    parser.set_defaults(autoload_proc=True)

    args = parser.parse_args()

    print(version_str)

    ops = {}
    if len(args.ops) > 0:
        ops = np.load(args.ops)
        ops = ops.item()
        if len(args.movie) > 0:
            process.run(args.movie, ops)

    # Always start by initializing Qt (only once per application)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "mouse.png"
    )
    app = QApplication(sys.argv)

    app_icon = QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)

    # Create and show the launch dialog
    launch_dialog = LaunchDialog()
    result = launch_dialog.exec_()

    if result == QDialog.Accepted:
        selected_option = launch_dialog.selected_option

        if selected_option == "Pose estimation":
            main_window = gui.run(
                args.movie,
                args.savedir,
                args.keypoints,
                args.proc_npy,
                args.neural_activity,
                args.neural_prediction,
                args.tneural,
                args.tbehavior,
                args.autoload_keypoints,
                args.autoload_proc,
            )
        elif selected_option == "Tongue segmentation":
            main_window = tongue_gui.run()
        
        main_window.show()

    elif result == QDialog.Rejected:  # Check if the dialog was canceled
        sys.exit(0)  # Exit the application when canceled

    sys.exit(app.exec_())
