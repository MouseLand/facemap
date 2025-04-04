"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
" Test imports for facemap package "


def test_facemap_imports():
    import facemap
    from facemap import process, pupil, running, utils


# def test_neural_prediction_imports():
#     from facemap.neural_prediction import (
#         neural_activity,
#         neural_model,
#         prediction_utils,
#     )


def test_pose_imports():
    from facemap.pose import (
        facemap_network,
        model_loader,
        pose,
        pose_helper_functions,
        transforms,
    )
