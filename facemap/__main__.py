import argparse
import time
from distutils.util import strtobool

import numpy as np

from facemap import process
from facemap.gui import gui


def tic():
    return time.time()


def toc(i0):
    return time.time() - i0


def main():
    ops = np.load("ops.npy")
    ops = ops.item()


# TODO: Add more description for the arguments
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
    parser.add_argument("--savedir", default=None, type=str, help="savedir")
    # Add a flag to autoload keypoints in the same directory as the movie
    parser.add_argument(
        "--autoload_keypoints",
        dest="autoload_keypoints",
        type=lambda x: bool(strtobool(x)),
        help="Automatically load keypoints in the same directory as the movie",
    )
    parser.set_defaults(autoload_keypoints=True)
    parser.add_argument(
        "--poseGUI",
        dest="poseGUI",
        action="store_true",
        help="Launch GUI w/ pose estimation",
    )
    parser.set_defaults(poseGUI=True)

    args = parser.parse_args()
    if args.poseGUI:
        print("Running Facemap w/ pose estimation GUI")
    else:
        print("Running Facemap w/o pose estimation GUI")

    ops = {}
    if len(args.ops) > 0:
        ops = np.load(args.ops)
        ops = ops.item()
        if len(args.movie) > 0:
            process.run(args.movie, ops)
    else:
        gui.run(
            args.movie,
            args.savedir,
            args.keypoints,
            args.neural_activity,
            args.neural_prediction,
            args.tneural,
            args.tbehavior,
            args.autoload_keypoints,
        )
