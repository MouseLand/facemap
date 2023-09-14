"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
from qtpy.QtWidgets import QAction
from qtpy.QtGui import QGuiApplication

from . import help_windows, io


def mainmenu(parent):
    # --------------- MENU BAR --------------------------
    open_file = QAction("Load video", parent)
    open_file.setShortcut("Ctrl+L")
    open_file.triggered.connect(lambda: io.open_file(parent))
    parent.addAction(open_file)

    open_folder = QAction("Load multiple videos", parent)
    open_folder.triggered.connect(lambda: io.open_folder(parent))
    parent.addAction(open_folder)

    # load processed data
    load_proc = QAction("Load *_proc.npy", parent)
    load_proc.triggered.connect(lambda: io.open_proc(parent))
    parent.addAction(load_proc)

    # Set output folder
    set_output_folder = QAction("Set output folder", parent)
    set_output_folder.setShortcut("Ctrl+S")
    set_output_folder.triggered.connect(lambda: io.save_folder(parent))
    parent.addAction(set_output_folder)

    load_pose = QAction("Load keypoints", parent)
    load_pose.setShortcut("Ctrl+K")
    load_pose.triggered.connect(lambda: io.get_pose_file(parent))
    parent.addAction(load_pose)

    train_model = QAction("Finetune model", parent)
    train_model.setShortcut("Ctrl+F")
    train_model.triggered.connect(lambda: parent.show_model_training_popup())
    parent.addAction(train_model)

    add_pose_model = QAction("Add pose model", parent)
    add_pose_model.triggered.connect(lambda: parent.add_pose_model())
    parent.addAction(add_pose_model)

    launch_neural_activity_window = QAction("Launch Neural Activity Window", parent)
    launch_neural_activity_window.triggered.connect(
        lambda: parent.launch_neural_activity_window()
    )
    parent.addAction(launch_neural_activity_window)
    """
    # Load neural data
    load_neural = QAction("Load neural data", parent)
    load_neural.triggered.connect(lambda: parent.load_neural_data())
    parent.addAction(load_neural)

    # Load neural predictions
    load_neural_predictions = QAction("Load neural predictions", parent)
    load_neural_predictions.triggered.connect(
        lambda: parent.load_neural_predictions_file()
    )
    parent.addAction(load_neural_predictions)

    # Run neural predictions
    run_neural_prediction = QAction("Run neural predictions", parent)
    run_neural_prediction.triggered.connect(
        lambda: parent.show_run_neural_predictions_dialog()
    )
    parent.addAction(run_neural_prediction)

    # Add a checkable action to toggle the visibility of test data in the predictions plot
    toggle_test_data = QAction("Highlight test data", parent)
    toggle_test_data.setCheckable(True)
    toggle_test_data.setChecked(True)
    toggle_test_data.triggered.connect(
        lambda: parent.toggle_testdata_display(toggle_test_data)
    )
    parent.addAction(toggle_test_data)
    """

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
    pose_menu.addAction(add_pose_model)

    neural_activity_menu = main_menu.addMenu("Neural activity")
    neural_activity_menu.addAction(launch_neural_activity_window)
    """"
    neural_activity_menu.addAction(load_neural)
    neural_activity_menu.addAction(load_neural_predictions)
    neural_activity_menu.addAction(run_neural_prediction)
    neural_activity_menu.addAction(toggle_test_data)
    """

    help_menu = main_menu.addMenu("&Help")
    help_menu.addAction(user_manual)
    help_menu.addAction(about_option)


def launch_user_manual(parent):
    help_windows.MainWindowHelp(parent, QGuiApplication.primaryScreen().availableGeometry())


def show_about(parent):
    help_windows.AboutWindow(parent, QGuiApplication.primaryScreen().availableGeometry())
