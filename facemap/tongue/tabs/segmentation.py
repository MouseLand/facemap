from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from ..tabs.videoplayer import VideoPlayer

class SegmentationTab(QWidget):
    def __init__(self):
        super().__init__()

        # Set up the UI
        self.setup_ui()

        # Initialize variables
        self.video_filenames = []

    def setup_ui(self):
        # Set up the layout
        self.layout = QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

        # Set up the splitter
        splitter = QSplitter(Qt.Horizontal)
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
        add_video_button = QPushButton("Add Video")
        add_video_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white;")
        add_video_button.clicked.connect(self.add_video)
        add_video_button.setStyleSheet(button_style)
        button_layout.addWidget(add_video_button)

        # Add number of videos loaded label
        self.num_videos_label = QLabel("0 Videos Loaded")
        self.num_videos_label.setStyleSheet("color: lightgrey;")
        self.num_videos_label.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(self.num_videos_label)

        show_filenames_button = QPushButton("Get loaded videos")
        show_filenames_button.setStyleSheet("background-color: rgb(196, 108, 57); color: white;")
        show_filenames_button.clicked.connect(self.show_video_filenames)
        show_filenames_button.setStyleSheet(button_style)
        button_layout.addWidget(show_filenames_button)

        # Add radio buttons for video views
        self.video_view_label = QLabel("Video View:")
        self.video_view_label.setAlignment(Qt.AlignCenter)
        self.video_view_label.setStyleSheet("color: lightgrey;")
        button_layout.addWidget(self.video_view_label)
        
        video_view_groupbox = QGroupBox()
        # change style sheet to remove border
        video_view_groupbox.setStyleSheet("border: none;")
        video_view_groupbox.setLayout(QHBoxLayout())
        self.video_view_group = QButtonGroup(button_panel)

        bottom_view_button = QRadioButton("Bottom")
        bottom_view_button.setStyleSheet("color: white;")
        self.video_view_group.addButton(bottom_view_button)
        video_view_groupbox.layout().addWidget(bottom_view_button)

        side_view_button = QRadioButton("Side")
        side_view_button.setStyleSheet("color: white;")
        self.video_view_group.addButton(side_view_button)
        video_view_groupbox.layout().addWidget(side_view_button)

        other_view_button = QRadioButton("Other")
        other_view_button.setStyleSheet("color: white;")
        self.video_view_group.addButton(other_view_button)
        video_view_groupbox.layout().addWidget(other_view_button)

        button_layout.addWidget(video_view_groupbox)

        """
        # Set up the right panel with loaded video label
        video_panel = QWidget()
        video_panel_layout = QVBoxLayout(video_panel)

        self.media_player = QMediaPlayer(self)
        self.video_widget = QVideoWidget(self)

        self.video_slider = QSlider(Qt.Horizontal)
        self.video_slider.setEnabled(False)
        self.video_slider.sliderMoved.connect(self.media_player.setPosition)        
        self.media_player.positionChanged.connect(self.update_slider_position)
        self.media_player.durationChanged.connect(self.video_slider.setMaximum)

        video_panel_layout.addWidget(self.video_widget)
        video_panel_layout.addWidget(self.video_slider)
        video_panel_layout.addStretch()

        # Set the size policy of the video panel
        video_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        """
        # Add the panels to the splitter
        splitter.addWidget(button_panel)
        splitter.addWidget(VideoPlayer())
        splitter.setSizes([100, 500])

        # Set the style sheet for the dark theme and use white text
        dark_stylesheet = """
            QWidget {
                background-color: rgb(50,50,50);
            }
            QSplitter::handle {
                background-color: rgb(80, 80, 80);
            }
        """
        self.setStyleSheet(dark_stylesheet)

        # Add the splitter to the layout
        self.layout.addWidget(splitter, 0, 0)

    def play_video(self, video_file):
        # Set the media player's media and set the video output to the video widget
        self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(video_file)))
        self.media_player.setVideoOutput(self.video_widget)

        # Add the video widget to the layout
        self.layout.addWidget(self.video_widget, 0, 1)

        # Enable the slider and set its maximum value to the media player's duration
        self.video_slider.setEnabled(True)
        self.video_slider.setMaximum(self.media_player.duration())

        # Play the video
        self.media_player.play()

    def show_video_filenames(self):
        # Show a message box with the list of video filenames
        if self.video_filenames:
            message = "\n".join(self.video_filenames)
        else:
            message = "No Videos Loaded"
        QMessageBox.information(self, "Loaded Videos", message)


    def add_video(self):
        # Show file dialog to select video files
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Video Files (*.mp4 *.avi *.mkv)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)
        if file_dialog.exec_():
            # Add the selected files to the list of video filenames
            self.video_filenames += file_dialog.selectedFiles()

            # Update the loaded videos label
            num_videos = len(self.video_filenames)
            if num_videos == 1:
                loaded_videos_text = "1 Video Loaded"
            else:
                loaded_videos_text = f"{num_videos} Videos Loaded"
            self.num_videos_label.setText(loaded_videos_text)

            # Play the first video in the list
            if num_videos == 1:
                self.play_video(self.video_filenames[0])

    def update_slider_position(self, position):
        self.video_slider.setValue(position)
