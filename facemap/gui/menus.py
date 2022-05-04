from PyQt5.QtWidgets import QAction, QDesktopWidget

from . import help_windows, io


def mainmenu(parent):
    # --------------- MENU BAR --------------------------
    # run suite2p from scratch
    openFile = QAction("Load single movie file", parent)
    openFile.triggered.connect(lambda: io.open_file(parent))
    parent.addAction(openFile)

    openFolder = QAction("Open folder of movies", parent)
    openFolder.setShortcut("Ctrl+O")
    openFolder.triggered.connect(lambda: io.open_folder(parent))
    parent.addAction(openFolder)

    # load processed data
    loadProc = QAction("&Load processed data", parent)
    loadProc.setShortcut("Ctrl+L")
    loadProc.triggered.connect(lambda: io.open_proc(parent))
    parent.addAction(loadProc)

    # Set output folder
    setOutputFolder = QAction("Set output folder", parent)
    setOutputFolder.triggered.connect(lambda: io.save_folder(parent))
    parent.addAction(setOutputFolder)

    # Load neural data
    loadNeural = QAction("Load neural data", parent)
    loadNeural.setShortcut("Ctrl+N")
    loadNeural.triggered.connect(lambda: parent.load_neural_data())
    parent.addAction(loadNeural)

    loadPose = QAction("Load keypoints", parent)
    # loadPose.setShortcut("Ctrl+P")
    loadPose.triggered.connect(lambda: io.get_pose_file(parent))
    parent.addAction(loadPose)

    train_model = QAction("&Train model", parent)
    train_model.setShortcut("Ctrl+T")
    train_model.triggered.connect(lambda: parent.show_model_training_popup())
    parent.addAction(train_model)

    load_finetuned_model = QAction("Load finetuned model", parent)
    load_finetuned_model.triggered.connect(lambda: parent.load_finetuned_model())
    parent.addAction(load_finetuned_model)

    user_manual = QAction("User manual", parent)
    user_manual.setShortcut("Ctrl+H")
    user_manual.triggered.connect(lambda: launch_user_manual(parent))
    parent.addAction(user_manual)

    about_option = QAction("About", parent)
    about_option.triggered.connect(lambda: show_about(parent))
    parent.addAction(about_option)

    # make mainmenu!
    main_menu = parent.menuBar()

    file_menu = main_menu.addMenu("&File")
    file_menu.grabShortcut("Ctrl+F")
    file_menu.addAction(openFile)
    file_menu.addAction(openFolder)
    file_menu.addAction(loadProc)
    file_menu.addAction(setOutputFolder)
    file_menu.addAction(loadNeural)

    pose_menu = main_menu.addMenu("Pose")
    pose_menu.addAction(loadPose)
    pose_menu.addAction(load_finetuned_model)
    pose_menu.addAction(train_model)

    help_menu = main_menu.addMenu("&Help")
    help_menu.addAction(user_manual)
    help_menu.addAction(about_option)


def launch_user_manual(parent):
    help_windows.MainWindowHelp(parent, QDesktopWidget().screenGeometry(-1))


def show_about(parent):
    help_windows.AboutWindow(parent, QDesktopWidget().screenGeometry(-1))
