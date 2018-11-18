import numpy as np
import time, os
from FaceMap import gui
from scipy import stats
import argparse

def tic():
    return time.time()
def toc(i0):
    return time.time() - i0

def main():
    ops = np.load('ops.npy')
    ops = ops.item()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Movie files')
    parser.add_argument('--ops', default=[], type=str, help='options')
    parser.add_argument('--movie', default=[], type=str, help='moviefile')
    args = parser.parse_args()

    ops = {}
    if len(args.ops)>0:
        ops = np.load(args.ops)
        ops = ops.item()
    if len(args.movie)>0:
        FaceMap.run(movie, ops)
    else:
        gui.run()
