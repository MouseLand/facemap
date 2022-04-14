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
    parser.add_argument("--movie", default=[], type=str, help="moviefile")
    parser.add_argument("--savedir", default=[], type=str, help="savedir")

    parser.add_argument(
        "--poseGUI", dest="poseGUI", action="store_true", help="Pose GUI"
    )
    parser.add_argument(
        "--no-poseGUI", dest="poseGUI", action="store_false", help="Pose CLI"
    )
    parser.set_defaults(poseGUI=True)

    args = parser.parse_args()

    if len(args.movie) > 0:
        moviefile = args.movie
    else:
        moviefile = None
    if len(args.savedir) > 0:
        savedir = args.savedir
    else:
        savedir = None

    if args.poseGUI:
        print("Running Facemap GUI w/ pose tracker")
    else:
        print("Running Facemap pose CLI")

    ops = {}
    if len(args.ops) > 0:
        ops = np.load(args.ops)
        ops = ops.item()
        if len(args.movie) > 0:
            process.run(args.movie, ops)
    else:
        gui.run(moviefile, savedir)
