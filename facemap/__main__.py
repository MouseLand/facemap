import argparse
import time

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Movie files")
    parser.add_argument("--ops", default=[], type=str, help="options")
    parser.add_argument("--movie", default=None, type=str, help="moviefile")
    parser.add_argument(
        "--keypoints",
        default=None,
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
    parser.add_argument("--savedir", default=None, type=str, help="savedir")
    parser.add_argument(
        "--poseGUI",
        dest="poseGUI",
        action="store_true",
        help="Launch GUI w/ pose estimation",
    )
    parser.set_defaults(poseGUI=True)

    args = parser.parse_args()

    # TODO: Add tags for loading tneural and tbehavior
    # FIXME: check loading and running batch files from CLI

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
        )
