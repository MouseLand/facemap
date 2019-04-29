import sys
import os
import shutil
import time
import numpy as np
from PyQt5 import QtGui, QtCore
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
import pims
from FaceMap import facemap, pupil
from scipy.stats import zscore, skew
from scipy.ndimage import gaussian_filter
from matplotlib import cm

colors = np.array([[0,200,50],[180,0,50],[40,100,250],[150,50,150]])

class sROI():
    def __init__(self, rind, rtype, iROI, moveable=True,
                 parent=None, saturation=None, color=None, pos=None,
                 yrange=None, xrange=None,
                 ivid=None):
        # what type of ROI it is
        self.iROI = iROI
        self.rind = rind
        self.rtype = rtype
        if saturation is None:
            self.saturation = 0
        else:
            self.saturation = saturation
        if color is None:
            self.color = np.maximum(0, np.minimum(255, colors[rind]+np.random.randn(3)*70))
            self.color = tuple(self.color)
        else:
            self.color = color
        self.pupil_sigma = 0
        self.moveable = moveable
        if pos is None:
            view = parent.p0.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, parent.Ly[0]*0.4)
            dy = np.minimum(dy, parent.Lx[0]*0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy = pos[0]
            imx = pos[1]
            dy = pos[2]
            dx = pos[3]
        if ivid is None:
            self.ivid=0
        else:
            self.ivid=ivid
            self.yrange=yrange
            self.xrange=xrange
        self.draw(parent, imy, imx, dy, dx)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))
        #self.position(parent)

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3,
                          style=QtCore.Qt.SolidLine)
        self.ROI = pg.RectROI(
            [imx, imy], [dx, dy], movable = self.moveable,
            pen=roipen, sideScalers=True, removable=self.moveable
        )
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0., 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.p0.addItem(self.ROI)

    def position(self, parent):
        parent.iROI = self.iROI
        pos0 = self.ROI.getSceneHandlePositions()
        pos = parent.p0.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        xrange = xrange[xrange >= 0]
        xrange = xrange[xrange < parent.LX]
        yrange = yrange[yrange >= 0]
        yrange = yrange[yrange < parent.LY]

        # which movie is this ROI in?
        vvals = parent.vmap[np.ix_(yrange, xrange)]
        ivid = np.zeros((len(parent.Ly),))
        for i in range(len(parent.Ly)):
            ivid[i] = (vvals==i).sum()
        ivid = np.argmax(ivid)
        # crop yrange and xrange
        xrange = xrange[xrange>=parent.sx[ivid]]
        xrange = xrange[xrange<parent.sx[ivid]+parent.Lx[ivid]]
        yrange = yrange[yrange>=parent.sy[ivid]]
        yrange = yrange[yrange<parent.sy[ivid]+parent.Ly[ivid]]

        xrange -= parent.sx[ivid]
        yrange -= parent.sy[ivid]
        self.xrange = xrange
        self.yrange = yrange
        self.ivid   = ivid

        parent.sl[1].setValue(parent.saturation[self.iROI] * 100 / 255)
        self.plot(parent)

    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        for i in range(len(parent.ROIs)):
            if i > self.iROI:
                parent.ROIs[i].iROI -= 1
                parent.saturation[i] = parent.saturation[i-1]
        del parent.ROIs[self.iROI]
        del parent.saturation[self.iROI]
        if parent.iROI>=len(parent.ROIs):
            parent.iROI -= 1
        parent.iROI = max(0, parent.iROI)
        parent.nROIs -= 1
        parent.pROIimg.clear()
        parent.pROI.removeItem(parent.scatter)
        parent.win.show()
        parent.show()

    def plot(self, parent):
        parent.iROI = self.iROI
        img = parent.imgs[self.ivid].copy()
        if img.ndim > 3:
            img = img.mean(axis=-2)
        img = img[np.ix_(self.yrange, self.xrange, np.arange(0,3))]
        sat = parent.saturation[self.iROI]
        self.saturation = sat

        self.pupil_sigma = parent.pupil_sigma

        if self.rind==0:
            img = img.mean(axis=-1)
            try:
                # smooth in space
                fr = gaussian_filter(img.astype(np.float32), 1)
                fr -= fr.min()
                fr = 255.0 - fr
                fr = np.maximum(0, fr - (255.0-sat))
                mu, sig, xy = pupil.fit_gaussian(fr, parent.pupil_sigma, True)
                xy = xy[xy[:,0]>=0, :]
                xy = xy[xy[:,0]<self.yrange.size, :]
                xy = xy[xy[:,1]>=0, :]
                xy = xy[xy[:,1]<self.xrange.size, :]
                parent.pROI.removeItem(parent.scatter)
                xy = np.concatenate((mu[np.newaxis,:], xy), axis=0)
                xy += 0.5

                parent.scatter = pg.ScatterPlotItem(xy[:,1], xy[:,0], pen=self.color, symbol='+')
                parent.pROI.addItem(parent.scatter)
                parent.pROIimg.setImage(255-fr)
                parent.pROIimg.setLevels([255-sat, 255])
            except:
                parent.pROI.removeItem(parent.scatter)
                parent.scatter = pg.ScatterPlotItem([0], [0], pen='k', symbol='+')
                parent.pROI.addItem(parent.scatter)
                parent.pROIimg.setImage(img)
                parent.pROIimg.setLevels([0, sat])
            parent.pROI.setRange(xRange=(0, self.xrange.size), yRange=(0,self.yrange.size))
        elif self.rind==1 or self.rind==3:
            parent.pROI.removeItem(parent.scatter)
            parent.scatter = pg.ScatterPlotItem([0], [0], pen='k', symbol='+')
            parent.pROI.addItem(parent.scatter)
            parent.pROIimg.setImage(img[:,:,1] - img[:,:,0])
            parent.pROIimg.setLevels([0, sat])
        elif self.rind==2:
            # blink
            img = img.mean(axis=-1)
            parent.pROI.removeItem(parent.scatter)
            parent.scatter = pg.ScatterPlotItem([0], [0], pen='k', symbol='+')
            parent.pROI.addItem(parent.scatter)
            fr  = img #- img.min()
            fr  = 255.0 - fr
            fr  = np.maximum(0, fr - (255.0-sat))
            parent.pROIimg.setImage(255-fr)
            parent.pROIimg.setLevels([255-sat, 255])
        parent.pROI.setRange(xRange=(0,img.shape[1]),
                         yRange=(0, img.shape[0]),
                          padding=0.0)
        parent.win.show()
        parent.show()
