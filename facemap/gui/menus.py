from PyQt5.QtWidgets import QAction, QDesktopWidget

from . import help_windows, io


def mainmenu(parent):
    # --------------- MENU BAR --------------------------
    # run suite2p from scratch
    open_file = QAction("Load single movie file", parent)
    open_file.triggered.connect(lambda: io.open_file(parent))
    parent.addAction(open_file)

    open_folder = QAction("Open folder of movies", parent)
    open_folder.setShortcut("Ctrl+O")
    open_folder.triggered.connect(lambda: io.open_folder(parent))
    parent.addAction(open_folder)

    # load processed data
    load_proc = QAction("&Load processed data", parent)
    load_proc.setShortcut("Ctrl+L")
    load_proc.triggered.connect(lambda: io.open_proc(parent))
    parent.addAction(load_proc)

    # Set output folder
    set_output_folder = QAction("Set output folder", parent)
    set_output_folder.triggered.connect(lambda: io.save_folder(parent))
    parent.addAction(set_output_folder)

    load_pose = QAction("Load keypoints", parent)
    load_pose.triggered.connect(lambda: io.get_pose_file(parent))
    parent.addAction(load_pose)

    train_model = QAction("&Train model", parent)
    train_model.setShortcut("Ctrl+T")
    train_model.triggered.connect(lambda: parent.show_model_training_popup())
    parent.addAction(train_model)

    # Load neural data
    load_neural = QAction("Load neural data", parent)
    load_neural.setShortcut("Ctrl+N")
    load_neural.triggered.connect(lambda: parent.load_neural_data())
    parent.addAction(load_neural)

    # Run neural predictions
    run_neural_prediction = QAction("Run neural predictions", parent)
    run_neural_prediction.setShortcut("Ctrl+R")
    run_neural_prediction.triggered.connect(lambda: parent.run_neural_predictions())
    parent.addAction(run_neural_prediction)

    # Load neural predictions
    load_neural_predictions = QAction("Load neural predictions", parent)
    load_neural_predictions.triggered.connect(
        lambda: parent.load_neural_data(prediction_mode=True)
    )

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
    file_menu.addAction(open_file)
    file_menu.addAction(open_folder)
    file_menu.addAction(load_proc)
    file_menu.addAction(set_output_folder)

    pose_menu = main_menu.addMenu("Pose")
    pose_menu.addAction(load_pose)
    pose_menu.addAction(train_model)

    neural_activity_menu = main_menu.addMenu("Neural activity")
    neural_activity_menu.addAction(load_neural)
    neural_activity_menu.addAction(run_neural_prediction)
    neural_activity_menu.addAction(load_neural_predictions)

    help_menu = main_menu.addMenu("&Help")
    help_menu.addAction(user_manual)
    help_menu.addAction(about_option)


def launch_user_manual(parent):
    help_windows.MainWindowHelp(parent, QDesktopWidget().screenGeometry(-1))


def show_about(parent):
    help_windows.AboutWindow(parent, QDesktopWidget().screenGeometry(-1))
