"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
" Test imports for facemap package "


def test_facemap_imports():
    import facemap
    from facemap import process, pupil, roi, running, utils


def test_gui_imports():
    from facemap.gui import cluster, gui, guiparts, help_windows, io, menus


def test_neural_prediction_imports():
    from facemap.neural_prediction import (
        neural_activity,
        neural_model,
        prediction_utils,
    )


def test_pose_imports():
    from facemap.pose import (
        facemap_network,
        model_loader,
        model_training,
        pose,
        pose_gui,
        pose_helper_functions,
        refine_pose,
        transforms,
    )
