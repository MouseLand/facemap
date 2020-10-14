import os, glob
import pims
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from . import guiparts, roi, utils
from natsort import natsorted

def open_file(parent, file_name=None):
    if file_name is None:
        file_name = QtGui.QFileDialog.getOpenFileName(parent,
                            "Open movie file")
    # load ops in same folder
    if file_name:
        print(file_name[0])
        parent.filelist = [ [file_name[0]] ]
        load_movies(parent)

def open_folder(parent, folder_name=None):
    if folder_name is None:
        folder_name = QtGui.QFileDialog.getExistingDirectory(parent,
                            "Choose folder with movies")
    # load ops in same folder
    if folder_name:
        extensions = ['*.mj2','*.mp4','*.mkv','*.avi','*.mpeg','*.mpg','*.asf']
        file_name = []
        for extension in extensions:
            files = glob.glob(folder_name+"/"+extension)
            files = [folder_name+"/"+os.path.split(f)[-1] for f in files]
            file_name.extend(files)
        for folder in glob.glob(folder_name+'/*/'):
            for extension in extensions:
                files = glob.glob(os.path.join(folder_name,folder,extension))
                files = [folder_name+"/"+folder+"/"+os.path.split(f)[-1] for f in files]
                file_name.extend(files)
        print(file_name)
        if len(file_name) > 1:
            choose_files(parent, file_name)
            load_movies(parent)

def choose_files(parent, file_name):
    parent.filelist = file_name
    LC=guiparts.ListChooser('Choose movies', parent)
    result = LC.exec_()
    if len(parent.filelist)==0:
        parent.filelist=file_name
    parent.filelist = natsorted(parent.filelist)
    if len(parent.filelist)>1:
        dm = QtGui.QMessageBox.question(
            parent,
            "multiple videos found",
            "are you processing multiple videos taken simultaneously?",
            QtGui.QMessageBox.Yes | QtGui.QMessageBox.No,
        )
        if dm == QtGui.QMessageBox.Yes:
            print('multi camera view')
            # expects first 4 letters to be different e.g. cam0, cam1, ...
            files = []
            iview = [os.path.basename(parent.filelist[0])[:4]]
            for f in parent.filelist[1:]:
                fbeg = os.path.basename(f)[:4]
                inview = np.array([iv==fbeg for iv in iview])
                if inview.sum()==0:
                    iview.append(fbeg)
            print(iview)
            for k in range(len(iview)):
                ij = 0
                for f in parent.filelist:
                    if iview[k] == os.path.basename(f)[:4]:
                        if k==0:
                            files.append([])
                        files[ij].append(f)
                        ij +=1
            parent.filelist = files
        else:
            print('single camera')
            files = parent.filelist.copy()
            parent.filelist = []
            for f in files:
                parent.filelist.append([f])

    else:
        parent.filelist = [parent.filelist]
    parent.filelist = natsorted(parent.filelist)
    print(parent.filelist)


def open_proc(parent, file_name=None):
    if file_name is None:
        file_name = QtGui.QFileDialog.getOpenFileName(parent,
                        "Open processed file", filter="*.npy")
        file_name = file_name[0]
    try:
        proc = np.load(file_name, allow_pickle=True).item()
        parent.filenames = proc['filenames']
        good=True
    except:
        good=False
        print("ERROR: not a processed movie file")
    if good:
        v = []
        nframes = 0
        iframes = []
        good = load_movies(parent, filelist=parent.filenames)
        if good:
            if 'fullSVD' in proc:
                parent.fullSVD = proc['fullSVD']
            else:
                parent.fullSVD = True
            k=0 # number of processed things
            parent.proctype = [0,0,0,0,0,0,0,0]
            parent.wroi = [0,0,0,0,0,0,0,0]

            if 'motSVD' in proc:
                parent.processed = True
            else:
                parent.processed = False

            iROI=0
            parent.typestr = ['pupil', 'motSVD', 'blink', 'run']
            if parent.processed:
                parent.col = []
                if parent.fullSVD:
                    parent.lbls[k].setText('fullSVD')
                    parent.lbls[k].setStyleSheet("color: white;")
                    parent.proctype[0] = 0
                    parent.col.append((255,255,255))
                    k+=1
                parent.motSVDs = proc['motSVD']
                parent.running = proc['running']
                parent.pupil = proc['pupil']
                parent.blink = proc['blink']
            else:
                k=0

            kt = [0,0,0,0]
            # whether or not you can move the ROIs
            moveable = not parent.processed
            if proc['rois'] is not None:
                for r in proc['rois']:
                    dy = r['yrange'][-1] - r['yrange'][0]
                    dx = r['xrange'][-1] - r['xrange'][0]
                    pos = [r['yrange'][0]+parent.sy[r['ivid']], r['xrange'][0]+parent.sx[r['ivid']], dy, dx]
                    parent.saturation.append(r['saturation'])
                    parent.rROI.append([])
                    parent.reflectors.append([])
                    if 'pupil_sigma' in r:
                        psig = r['pupil_sigma']
                        parent.pupil_sigma = psig
                        parent.sigmaBox.setText(str(r['pupil_sigma']))
                    else:
                        psig = None
                    parent.ROIs.append(roi.sROI(rind=r['rind'], rtype=r['rtype'], iROI=r['iROI'], color=r['color'],
                                        moveable=moveable, parent=parent, saturation=r['saturation'], pupil_sigma=psig,
                                        yrange=r['yrange'], xrange=r['xrange'], pos=pos, ivid=r['ivid']))
                    if 'reflector' in r:
                        for i,rr in enumerate(r['reflector']):
                            pos = [rr['yrange'][0], rr['xrange'][0], rr['yrange'][-1]-rr['yrange'][0], rr['xrange'][-1]-rr['xrange'][0]]
                            parent.rROI[-1].append(roi.reflectROI(iROI=r['iROI'], wROI=i, pos=pos, parent=parent,
                                                yrange=rr['yrange'], xrange=rr['xrange'], ellipse=rr['ellipse']))
                    if parent.fullSVD:
                        parent.iROI = k-1
                    else:
                        parent.iROI = k
                    parent.ROIs[-1].ellipse = r['ellipse']
                    #parent.ROIs[-1].position(parent)
                    parent.sl[1].setValue(parent.saturation[parent.iROI] * 100 / 255)
                    parent.ROIs[parent.iROI].plot(parent)
                    if parent.processed:
                        if k < 8:
                            parent.lbls[k].setText('%s%d'%(parent.typestr[r['rind']], kt[r['rind']]))
                            parent.lbls[k].setStyleSheet("color: rgb(%s,%s,%s);"%
                                                        (str(int(r['color'][0])), str(int(r['color'][1])), str(int(r['color'][2]))))
                            parent.wroi[k] = kt[r['rind']]
                            kt[r['rind']]+=1
                            parent.proctype[k] = r['rind'] + 1
                            parent.col.append(r['color'])
                    k+=1
            parent.kroi = k

            # initialize plot
            parent.cframe = 1
            if parent.processed:
                for k in range(parent.kroi):
                    parent.cbs1[k].setEnabled(True)
                    parent.cbs2[k].setEnabled(True)
                if parent.fullSVD:
                    parent.cbs1[0].setChecked(True)
                parent.plot_processed()

            parent.next_frame()

def load_movies(parent, filelist=None):
    if filelist is not None:
        parent.filelist = filelist
    try:
        v = []
        nframes = 0
        iframes = []
        cumframes = [0]
        k=0
        for fs in parent.filelist:
            vs = []
            for f in fs:
                try:
                    vs.append(pims.Video(f))
                except:
                    print('pyavreaderindexed used - may be slower (try installing pims github version)')
                    vs.append(pims.PyAVReaderIndexed(f))
            v.append(vs)
            iframes.append(len(v[-1][0]))
            cumframes.append(cumframes[-1] + len(v[-1][0]))
            nframes += len(v[-1][0])
            if k==0:
                Ly = []
                Lx = []
                for vs in v[-1]:
                    fshape = vs.frame_shape
                    Ly.append(fshape[0])
                    Lx.append(fshape[1])
            k+=1
        good = True
    except Exception as e:
        print("ERROR: not a supported movie file")
        print(e)
        good = False
    if good:
        parent.reset()
        parent.video = v
        parent.filenames = parent.filelist
        parent.nframes = nframes
        parent.iframes = np.array(iframes).astype(int)
        parent.cumframes = np.array(cumframes).astype(int)
        parent.Ly = Ly
        parent.Lx = Lx
        parent.p1.clear()
        parent.p2.clear()
        if len(parent.Ly)<2:
            parent.LY = parent.Ly[0]
            parent.LX = parent.Lx[0]
            parent.sx = np.array([int(0)])
            parent.sy = np.array([int(0)])
            parent.vmap = np.zeros((parent.LY,parent.LX), np.int32)
        else:
            # make placement of movies
            Ly = np.array(parent.Ly.copy())
            Lx = np.array(parent.Lx.copy())

            LY, LX, sy, sx = utils.video_placement(Ly, Lx)
            print(LY, LX)
            parent.vmap = -1 * np.ones((LY,LX), np.int32)
            for i in range(Ly.size):
                parent.vmap[sy[i]:sy[i]+Ly[i], sx[i]:sx[i]+Lx[i]] = i
            parent.sy = sy
            parent.sx = sx
            parent.LY = LY
            parent.LX = LX

        parent.fullimg = np.zeros((parent.LY, parent.LX, 3))
        parent.imgs = []
        parent.img = []
        for i in range(len(parent.Ly)):
            parent.imgs.append(np.zeros((parent.Ly[i], parent.Lx[i], 3, 3)))
            parent.img.append(np.zeros((parent.Ly[i], parent.Lx[i], 3)))
        parent.movieLabel.setText(os.path.dirname(parent.filenames[0][0]))
        parent.frameDelta = int(np.maximum(5,parent.nframes/200))
        parent.frameSlider.setSingleStep(parent.frameDelta)
        if parent.nframes > 0:
            parent.updateFrameSlider()
            parent.updateButtons()
        parent.cframe = 1
        parent.loaded = True
        parent.processed = False
        parent.jump_to_frame()
    return good
