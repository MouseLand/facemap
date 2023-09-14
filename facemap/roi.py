"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore

from facemap import pupil, utils

# Types of ROI and their ID:
# 0: Pupil
# 1: motion SVD
# 2: Blink
# 3: Running
# 4: Pose bbox
colors = np.array(
    [[0, 200, 50], [180, 0, 50], [40, 100, 250], [150, 50, 150], [0, 255, 255]]
)


class reflectROI:
    def __init__(
        self,
        iROI,
        wROI,
        moveable=True,
        parent=None,
        pos=None,
        yrange=None,
        xrange=None,
        ellipse=None,
    ):
        # which ROI it belongs to
        self.iROI = iROI
        self.wROI = wROI  # can have many reflections
        self.color = (0.0, 0.0, 0.0)
        self.moveable = moveable
        if pos is None:
            view = parent.pROI.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, parent.Ly[0] * 0.4)
            dy = np.minimum(dy, parent.Lx[0] * 0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy = pos[0]
            imx = pos[1]
            dy = pos[2]
            dx = pos[3]
            self.yrange = yrange
            self.xrange = xrange
            self.ellipse = ellipse
        self.ivid = parent.ROIs[self.iROI].ivid
        self.draw(parent, imy, imx, dy, dx)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))
        # self.position(parent)

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3, style=QtCore.Qt.SolidLine)
        self.ROI = pg.EllipseROI(
            [imx, imy],
            [dx, dy],
            movable=self.moveable,
            pen=roipen,
            removable=self.moveable,
        )
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0.0, 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.pROI.addItem(self.ROI)

    def remove(self, parent):
        parent.pROI.removeItem(self.ROI)
        for i in range(len(parent.rROI[self.iROI])):
            if i > self.wROI:
                parent.rROI[self.iROI][i].wROI -= 1
        del parent.rROI[self.iROI][self.wROI]
        parent.roi_embed_window.show()
        parent.show()

    def position(self, parent):
        parent.iROI = self.iROI
        pos0 = self.ROI.getSceneHandlePositions()
        pos = parent.pROI.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        yrange += int(sizey / 2)
        # what is ellipse circling?
        br = self.ROI.boundingRect()
        ellipse = np.zeros((yrange.size, xrange.size), "bool")
        x, y = np.meshgrid(np.arange(0, xrange.size, 1), np.arange(0, yrange.size, 1))
        ellipse = (
            (y - br.center().y()) ** 2 / (br.height() / 2) ** 2
            + (x - br.center().x()) ** 2 / (br.width() / 2) ** 2
        ) <= 1

        ellipse = ellipse[
            :, np.logical_and(xrange >= 0, xrange < parent.ROIs[self.iROI].xrange.size)
        ]
        xrange = xrange[
            np.logical_and(xrange >= 0, xrange < parent.ROIs[self.iROI].xrange.size)
        ]
        ellipse = ellipse[
            np.logical_and(yrange >= 0, yrange < parent.ROIs[self.iROI].yrange.size), :
        ]
        yrange = yrange[
            np.logical_and(yrange >= 0, yrange < parent.ROIs[self.iROI].yrange.size)
        ]

        # ellipse = lambda x,y: (((x+0.5)/(w/2.)-1)**2+ ((y+0.5)/(h/2.)-1)**2)**0.5 < 1, (w, h))
        self.ellipse = ellipse
        self.xrange = xrange
        self.yrange = yrange

        parent.reflectors[self.iROI] = utils.get_reflector(
            parent.ROIs[self.iROI].yrange,
            parent.ROIs[self.iROI].xrange,
            parent.rROI[self.iROI],
        )

        parent.saturation_sliders[1].setValue(
            int(parent.saturation[self.iROI] * 100 / 255)
        )
        parent.ROIs[self.iROI].plot(parent)


class sROI:
    def __init__(
        self,
        rind,
        rtype,
        iROI,
        moveable=True,
        resizable=True,
        parent=None,
        saturation=None,
        color=None,
        pos=None,
        yrange=None,
        xrange=None,
        ivid=None,
        pupil_sigma=None,
    ):
        # what type of ROI it is
        self.iROI = iROI
        self.rind = rind
        self.rtype = rtype
        self.pos = pos
        if saturation is None:
            self.saturation = 0
        else:
            self.saturation = saturation
        if color is None:
            self.color = np.maximum(
                0, np.minimum(255, colors[rind] + np.random.randn(3) * 70)
            )
            self.color = tuple(self.color)
        else:
            self.color = color
        if pupil_sigma is not None:
            self.pupil_sigma = pupil_sigma
        else:
            self.pupil_sigma = 0
        self.moveable = moveable
        self.resizable = resizable
        if self.pos is None:
            view = parent.p0.viewRange()
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, parent.Ly[0] * 0.4)
            dy = np.minimum(dy, parent.Lx[0] * 0.4)
            imx = imx - dx / 2
            imy = imy - dy / 2
        else:
            imy = self.pos[0]
            imx = self.pos[1]
            dy = self.pos[2]
            dx = self.pos[3]
        if ivid is None:
            self.ivid = 0
        else:
            self.ivid = ivid
            self.yrange = yrange
            self.xrange = xrange
        self.draw(parent, imy, imx, dy, dx)
        self.ROI.sigRegionChangeFinished.connect(lambda: self.position(parent))
        self.ROI.sigClicked.connect(lambda: self.position(parent))
        self.ROI.sigRemoveRequested.connect(lambda: self.remove(parent))
        # self.position(parent)

    def draw(self, parent, imy, imx, dy, dx):
        roipen = pg.mkPen(self.color, width=3, style=QtCore.Qt.SolidLine)
        if self.rind == 1 or self.rind == 3 or self.rind == 4:
            self.ROI = pg.RectROI(
                [imx, imy],
                [dx, dy],
                movable=self.moveable,
                resizable=self.resizable,
                pen=roipen,
                sideScalers=True,
                removable=self.moveable,
            )
        else:
            self.ROI = pg.EllipseROI(
                [imx, imy],
                [dx, dy],
                movable=self.moveable,
                resizable=self.resizable,
                pen=roipen,
                removable=self.moveable,
            )
        self.ROI.handleSize = 8
        self.ROI.handlePen = roipen
        self.ROI.addScaleHandle([1, 0.5], [0.0, 0.5])
        self.ROI.addScaleHandle([0.5, 0], [0.5, 1])
        self.ROI.setAcceptedMouseButtons(QtCore.Qt.LeftButton)
        parent.p0.addItem(self.ROI)

    def position(self, parent):
        if parent.iROI != self.iROI:
            if self.rind == 0:
                # print('change to pupil')
                for i in range(len(parent.rROI[self.iROI])):
                    parent.pROI.addItem(parent.rROI[self.iROI][i].ROI)
            elif parent.ROIs[parent.iROI].rind == 0:
                for i in range(len(parent.rROI[parent.iROI])):
                    parent.pROI.removeItem(parent.rROI[parent.iROI][i].ROI)
            parent.iROI = self.iROI
        pos0 = self.ROI.getSceneHandlePositions()
        pos = parent.p0.mapSceneToView(pos0[0][1])
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        self.pos = posy, posx, posy + sizey, posx + sizex
        # self.pos = (posy, posx, posy+sizey, posx+sizex) # get ROI position
        if self.rind == 0 or self.rind == 2:
            yrange += int(sizey / 2)
        # what is ellipse circling?
        br = self.ROI.boundingRect()
        ellipse = np.zeros((yrange.size, xrange.size), "bool")
        x, y = np.meshgrid(np.arange(0, xrange.size, 1), np.arange(0, yrange.size, 1))
        ellipse = (
            (y - br.center().y()) ** 2 / (br.height() / 2) ** 2
            + (x - br.center().x()) ** 2 / (br.width() / 2) ** 2
        ) <= 1

        ellipse = ellipse[:, np.logical_and(xrange >= 0, xrange < parent.LX)]
        xrange = xrange[np.logical_and(xrange >= 0, xrange < parent.LX)]
        ellipse = ellipse[np.logical_and(yrange >= 0, yrange < parent.LY), :]
        yrange = yrange[np.logical_and(yrange >= 0, yrange < parent.LY)]

        # ellipse = lambda x,y: (((x+0.5)/(w/2.)-1)**2+ ((y+0.5)/(h/2.)-1)**2)**0.5 < 1, (w, h))
        # which movie is this ROI in?
        vvals = parent.vmap[np.ix_(yrange, xrange)]
        ivid = np.zeros((len(parent.Ly),))
        for i in range(len(parent.Ly)):
            ivid[i] = (vvals == i).sum()
        ivid = np.argmax(ivid)
        # crop yrange and xrange
        ix = np.logical_and(
            xrange >= parent.sx[ivid], xrange < parent.sx[ivid] + parent.Lx[ivid]
        )
        ellipse = ellipse[:, ix]
        xrange = xrange[ix]
        iy = np.logical_and(
            yrange >= parent.sy[ivid], yrange < parent.sy[ivid] + parent.Ly[ivid]
        )
        yrange = yrange[iy]
        ellipse = ellipse[iy, :]
        self.ellipse = ellipse

        xrange -= parent.sx[ivid]
        yrange -= parent.sy[ivid]
        self.xrange = xrange
        self.yrange = yrange
        self.ivid = ivid

        if self.rind == 0:
            self.rmin = 0
            parent.reflectors[self.iROI] = utils.get_reflector(
                parent.ROIs[self.iROI].yrange,
                parent.ROIs[self.iROI].xrange,
                rROI=parent.rROI[self.iROI],
            )
        parent.saturation_sliders[1].setValue(
            int(parent.saturation[self.iROI] * 100 / 255)
        )

        index = parent.roi_embed_combobox.findText("ROI", QtCore.Qt.MatchFixedString)
        if index >= 0:
            parent.roi_embed_combobox.setCurrentIndex(index)
        parent.roi_display_combobox.setCurrentIndex(self.iROI + 1)
        parent.display_ROI()  # self.plot(parent)

    def remove(self, parent):
        parent.p0.removeItem(self.ROI)
        for i in range(len(parent.ROIs)):
            if i > self.iROI:
                parent.ROIs[i].iROI -= 1
                parent.saturation[i] = parent.saturation[i - 1]
        del parent.ROIs[self.iROI]
        del parent.saturation[self.iROI]
        del parent.rROI[self.iROI]
        if parent.iROI >= len(parent.ROIs):
            parent.iROI -= 1
        parent.iROI = max(0, parent.iROI)
        parent.nROIs -= 1
        parent.pROIimg.clear()
        parent.pROI.removeItem(parent.scatter)
        parent.roi_embed_window.show()
        parent.show()
        parent.update_ROI_vis_comboBox()

    def plot(self, parent):
        parent.iROI = self.iROI
        img = parent.imgs[self.ivid].copy()
        if img.ndim > 3:
            img = img.mean(axis=-2)
        img = img[
            self.yrange[0] : self.yrange[-1] + 1, self.xrange[0] : self.xrange[-1] + 1
        ]
        sat = parent.saturation[self.iROI]
        self.saturation = sat
        parent.set_ROI_saturation_label(sat * 100 / 255)

        self.pupil_sigma = parent.pupil_sigma
        # parent.pROI.addItem(pg.ScatterPlotItem([self.center[0]], [self.center[1]], pen='r', symbol='+'))
        parent.reflector.setEnabled(False)
        if self.rind == 0:
            parent.reflector.setEnabled(True)
            if img.ndim > 2:
                img = img.mean(axis=-1)
            # smooth in space
            fr = img.astype(np.float32)
            # fr = gaussian_filter(img.astype(np.float32), 1)
            # fr -= self.rmin
            fr[~self.ellipse] = 255.0
            fr = 255.0 - fr
            fr = np.maximum(0, fr - (255.0 - sat))
            missing = parent.reflectors[self.iROI]
            try:
                mu, sig, _, _, xy, immiss = pupil.fit_gaussian(
                    fr.copy(), parent.pupil_sigma, do_xy=True, missing=missing
                )
                area = np.pi * (sig[0] * sig[1]) ** 0.5
                if len(missing) > 0:
                    fr[missing[0], missing[1]] = immiss
                xy = xy[xy[:, 0] >= 0, :]
                xy = xy[xy[:, 0] < self.yrange.size, :]
                xy = xy[xy[:, 1] >= 0, :]
                xy = xy[xy[:, 1] < self.xrange.size, :]
                parent.pROI.removeItem(parent.scatter)
                xy = np.concatenate((mu[np.newaxis, :], xy), axis=0)
                xy += 0.5
                pen = pg.mkPen(self.color, width=2)
                parent.scatter = pg.ScatterPlotItem(
                    xy[:, 1], xy[:, 0], pen=pen, symbol="+"
                )
                parent.pROI.addItem(parent.scatter)
                parent.pROIimg.setImage(255 - fr)
                parent.pROIimg.setLevels([255 - sat, 255])
            except:
                print("no pupil found")
                parent.pROI.removeItem(parent.scatter)
                parent.scatter = pg.ScatterPlotItem([0], [0], pen="k", symbol="+")
                parent.pROI.addItem(parent.scatter)
                parent.pROIimg.setImage(255 - fr)
                parent.pROIimg.setLevels([255 - sat, 255])
                area = np.nan
                mu = [np.nan, np.nan]
            parent.pROI.setRange(
                xRange=(0, self.xrange.size), yRange=(0, self.yrange.size)
            )
            if parent.online_mode:
                if parent.online_traces is None:
                    parent.online_traces = np.zeros((3, 0))
                if parent.playButton.isChecked() and not parent.online_plotted:
                    parent.online_traces = np.append(
                        parent.online_traces,
                        np.array([mu[0], mu[1], area])[:, np.newaxis],
                        axis=1,
                    )
                    traces = parent.online_traces.copy()
                    nframes = traces.shape[-1]
                    if nframes > 1:
                        traces -= traces.mean(axis=-1)[:, np.newaxis]
                        norm = traces.std(axis=-1)
                        norm[np.logical_or(norm == 0, np.isnan(norm))] = 1.0
                        parent.keypoints_traces_plot.clear()
                        pen = pg.mkPen(self.color, width=2)
                        parent.keypoints_traces_plot.plot(
                            traces[0] / norm[0] * 2, pen=pen
                        )
                        pen = pg.mkPen(
                            (155, 255, 155), width=1, style=QtCore.Qt.DashLine
                        )
                        parent.keypoints_traces_plot.plot(
                            traces[1] / norm[1] * 2, pen=pen
                        )
                        pen = pg.mkPen((0, 100, 0), width=1, style=QtCore.Qt.DashLine)
                        parent.keypoints_traces_plot.plot(
                            traces[2] / norm[2] * 2, pen=pen
                        )
                        parent.keypoints_traces_plot.setRange(
                            xRange=(0, nframes), yRange=(-4, 4), padding=0.0
                        )
                        parent.keypoints_traces_plot.setLimits(xMin=0, xMax=nframes)
                        parent.keypoints_traces_plot.show()
                        parent.roi_embed_window.show()
                        parent.show()
                        parent.online_plotted = True
                # self.svd_traces_plot.setLimits(xMin=0,xMax=self.nframes)
        elif self.rind == 1 or self.rind == 3 or self.rind == 4:
            parent.pROI.removeItem(parent.scatter)
            parent.scatter = pg.ScatterPlotItem([0], [0], pen="k", symbol="+")
            parent.pROI.addItem(parent.scatter)
            parent.pROIimg.setImage(img[:, :, 1] - img[:, :, 0])
            parent.pROIimg.setLevels([0, sat])
        elif self.rind == 2:
            # blink
            img = img.mean(axis=-1)
            parent.pROI.removeItem(parent.scatter)
            parent.scatter = pg.ScatterPlotItem([0], [0], pen="k", symbol="+")
            parent.pROI.addItem(parent.scatter)
            fr = img  # - img.min()
            fr = 255.0 - fr
            fr = np.maximum(0, fr - (255.0 - sat))
            parent.pROIimg.setImage(255 - fr)
            parent.pROIimg.setLevels([255 - sat, 255])
        if parent.roi_embed_combobox.currentText() != "ROI":
            parent.pROIimg.clear()
        parent.pROI.setRange(
            xRange=(0, img.shape[1]), yRange=(0, img.shape[0]), padding=0.0
        )
        parent.roi_embed_window.show()
        parent.show()
