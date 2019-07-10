import numpy as np
import time, os
from facemap import gui,process
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
    parser.add_argument('--savedir', default=[], type=str, help='savedir')
    args = parser.parse_args()

    if len(args.movie)>0:
        moviefile = args.movie
    else:
        moviefile = None
    if len(args.savedir)>0:
        savedir = args.savedir
    else:
        savedir = None

    ops = {}
    if len(args.ops)>0:
        ops = np.load(args.ops)
        ops = ops.item()
        if len(args.movie)>0:
            process.run(args.movie, ops)
    else:
        gui.run(moviefile, savedir)
        
