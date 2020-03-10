import glob, os, time
from PyQt5 import QtTest
import numpy as np
import cv2
from . import guiparts

def normalize99(img):
    X = img.copy()
    X = (X - np.percentile(X, 1)) / (np.percentile(X, 99) - np.percentile(X, 1))
    return X

def choose_folder(parent):
    TC = guiparts.TextChooser(parent)
    result = TC.exec_()
    if result:
        parent.folder = TC.folder
        parent.movieLabel.setText(parent.folder)
    else:
        return
    parent.irand = 0 
    parent.proc = 0 
    parent.flag = 1
    parent.online_mode = True
    
    # enable certain buttons
    parent.frameSlider.setEnabled(False)
    parent.playButton.setEnabled(True)
    parent.pauseButton.setEnabled(False)
    parent.addROI.setEnabled(True)
    parent.pauseButton.setChecked(True)

    parent.online_traces = None
    parent.Ly = [224]
    parent.Lx = [224]
    parent.LY = 224
    parent.LX = 224
    parent.sy = [0]
    parent.sx = [0]
    parent.vmap = np.zeros((224,224), np.int32)
    parent.next_frame()
    #parent.updateTimer.start(0.001)

def get_frame(parent):
    while 1:
        fsnew = glob.glob(os.path.join(parent.folder, 'frame%d.png'%parent.irand))
        if len(fsnew)==1:
            parent.flag = 0
            parent.irand += 1
            fs = fsnew
        else:
            if parent.flag==1:
                #time.sleep(.01)
                QtTest.QTest.qWait(100)
            else:
                parent.flag = 1
                break

    print('found frame %d'%(parent.irand-1))
    #parent.file_name.setText('frame%d.png'%(parent.irand-1))

    frame = cv2.imread(fs[0])

    eye = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    eye = cv2.resize(eye, (224,224))
    img = np.float32(eye)
    parent.fullimg = normalize99(img) * 255.
    parent.imgs = [parent.fullimg]
    parent.cframe = parent.irand-1
