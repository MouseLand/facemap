"""
Copright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Atika Syeda.
"""
import numpy as np
import pyqtgraph as pg
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QDialog,
    QGraphicsPathItem,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSlider,
    QStyleOptionSlider,
    QWidget,
)
from pyqtgraph import Point


## Following is adapted from https://stackoverflow.com/a/17108463 for a faster implementation of the multiline plot for neural activity
class MultiLine(QGraphicsPathItem):
    def __init__(self, x, y):
        """
        Multiline class for plotting 2D data as a series of lines.
        Parameters
        ----------
        x : 2D-array
            array of shape (Nplots, Nsamples)
        y : 2D-array
            array of shape (Nplots, Nsamples)
        """
        connect = np.ones(x.shape, dtype=bool)
        connect[:, -1] = 0  # don't draw the segment between each trace
        self.path = pg.arrayToQPath(x.flatten(), y.flatten(), connect.flatten())
        QGraphicsPathItem.__init__(self, self.path)
        self.setPen(pg.mkPen("w"))

    def shape(self):  # override because QGraphicsPathItem.shape is too expensive.
        return QGraphicsItem.shape(self)

    def boundingRect(self):
        return self.path.boundingRect()


### custom QDialog which makes a list of items you can include/exclude
class ListChooser(QDialog):
    def __init__(self, title, parent):
        super(ListChooser, self).__init__(parent)
        self.setGeometry(300, 300, np.floor(parent.sizeObject.width() * 0.35).astype(int), np.floor(parent.sizeObject.height() * 0.22).astype(int))
        self.setMinimumHeight(np.floor(parent.sizeObject.height() * 0.22).astype(int))
        self.setMinimumWidth(np.floor(parent.sizeObject.width() * 0.35).astype(int))
        self.setWindowTitle(title)
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        layout.addWidget(
            QLabel("click to select videos (none selected => all used)"), 0, 0, 1, 1
        )
        self.list = QListWidget(parent)
        self.list.setMinimumWidth(np.floor(parent.sizeObject.width() * 0.325).astype(int))
        self.list.setMinimumHeight(np.floor(parent.sizeObject.height() * 0.15).astype(int))
        for f in parent.filelist:
            self.list.addItem(f)
        layout.addWidget(self.list, 1, 0, 7, 1)
        self.list.setSelectionMode(QAbstractItemView.MultiSelection)
        done = QPushButton("done")
        done.clicked.connect(lambda: self.exit_list(parent))
        layout.addWidget(done, 8, 0, 1, 1)

    def exit_list(self, parent):
        parent.filelist = []
        items = self.list.selectedItems()
        for i in range(len(items)):
            parent.filelist.append(str(self.list.selectedItems()[i].text()))
        self.accept()


class Slider(QSlider):
    def __init__(self, bid, parent=None):
        super(self.__class__, self).__init__()
        initval = [99, 99]
        self.bid = bid
        self.setOrientation(QtCore.Qt.Horizontal)
        self.setMinimum(0)
        self.setMaximum(100)
        self.setValue(initval[bid])
        self.setTickInterval(10)
        self.valueChanged.connect(lambda: self.level_change(parent, bid))
        self.setTracking(False)

    def level_change(self, parent, bid):
        parent.sat[bid] = float(self.value()) / 100 * 255
        if bid == 0:
            parent.pimg.setLevels([0, parent.sat[bid]])
        else:
            # parent.pROIimg.setLevels([0, parent.sat[bid]])
            parent.saturation[parent.iROI] = parent.sat[bid]
            if len(parent.ROIs) > 0:
                parent.ROIs[parent.iROI].plot(parent)
        parent.roi_embed_window.show()


class TextChooser(QDialog):
    def __init__(self, parent=None):
        super(TextChooser, self).__init__(parent)
        self.setGeometry(300, 300, 350, 100)
        self.setWindowTitle("folder path")
        self.win = QWidget(self)
        layout = QGridLayout()
        self.win.setLayout(layout)
        self.qedit = QLineEdit("")
        layout.addWidget(QLabel("folder name (does not have to exist yet)"), 0, 0, 1, 3)
        layout.addWidget(self.qedit, 1, 0, 1, 3)
        done = QPushButton("OK")
        done.clicked.connect(self.exit)
        layout.addWidget(done, 2, 1, 1, 1)

    def exit(self):
        self.folder = self.qedit.text()
        self.accept()


class RGBRadioButtons(QButtonGroup):
    def __init__(self, parent=None, row=0, col=0):
        super(RGBRadioButtons, self).__init__()
        parent.color = 0
        self.parent = parent
        self.bstr = ["image", "flowsX", "flowsY", "flowsZ", "cellprob"]
        self.dropdown = []
        for b in range(len(self.bstr)):
            button = QRadioButton(self.bstr[b])
            button.setStyleSheet("color: white;")
            if b == 0:
                button.setChecked(True)
            self.addButton(button, b)
            button.toggled.connect(lambda: self.btnpress(parent))
            self.parent.scene_grid_layout.addWidget(button, row + b, col, 1, 1)
        self.setExclusive(True)
        # self.buttons.

    def btnpress(self, parent):
        b = self.checkedId()
        self.parent.view = b
        if self.parent.loaded:
            self.parent.update_plot()


class ViewBoxNoRightDrag(pg.ViewBox):
    def __init__(
        self,
        parent=None,
        border=None,
        lockAspect=False,
        enableMouse=True,
        invertY=False,
        enableMenu=True,
        name=None,
        invertX=False,
    ):
        pg.ViewBox.__init__(
            self,
            parent,
            border,
            lockAspect,
            enableMouse,
            invertY,
            enableMenu,
            name,
            invertX,
        )

    def mouseDragEvent(self, ev, axis=None):
        ## if axis is specified, event will only affect that axis.
        ev.accept()  ## we accept all buttons

        pos = ev.pos()
        lastPos = ev.lastPos()
        dif = pos - lastPos
        dif = dif * -1

        ## Ignore axes if mouse is disabled
        mouseEnabled = np.array(self.state["mouseEnabled"], dtype=np.float)
        mask = mouseEnabled.copy()
        if axis is not None:
            mask[1 - axis] = 0.0

        ## Scale or translate based on mouse button
        if ev.button() & (QtCore.Qt.LeftButton | QtCore.Qt.MidButton):
            if self.state["mouseMode"] == pg.ViewBox.RectMode:
                if (
                    ev.isFinish()
                ):  ## This is the final move in the drag; change the view scale now
                    # print "finish"
                    self.rbScaleBox.hide()
                    ax = QtCore.QRectF(Point(ev.buttonDownPos(ev.button())), Point(pos))
                    ax = self.childGroup.mapRectFromParent(ax)
                    self.showAxRect(ax)
                    self.axHistoryPointer += 1
                    self.axHistory = self.axHistory[: self.axHistoryPointer] + [ax]
                else:
                    ## update shape of scale box
                    self.updateScaleBox(ev.buttonDownPos(), ev.pos())
            else:
                tr = dif * mask
                tr = self.mapToView(tr) - self.mapToView(Point(0, 0))
                x = tr.x() if mask[0] == 1 else None
                y = tr.y() if mask[1] == 1 else None

                self._resetTarget()
                if x is not None or y is not None:
                    self.translateBy(x=x, y=y)
                self.sigRangeChangedManually.emit(self.state["mouseEnabled"])


class ImageDraw(pg.ImageItem):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`
    GraphicsObject displaying an image. Optimized for rapid update (ie video display).
    This item displays either a 2D numpy array (height, width) or
    a 3D array (height, width, RGBa). This array is optionally scaled (see
    :func:`setLevels <pyqtgraph.ImageItem.setLevels>`) and/or colored
    with a lookup table (see :func:`setLookupTable <pyqtgraph.ImageItem.setLookupTable>`)
    before being displayed.
    ImageItem is frequently used in conjunction with
    :class:`HistogramLUTItem <pyqtgraph.HistogramLUTItem>` or
    :class:`HistogramLUTWidget <pyqtgraph.HistogramLUTWidget>` to provide a GUI
    for controlling the levels and lookup table used to display the image.
    """

    sigImageChanged = QtCore.Signal()

    def __init__(self, image=None, viewbox=None, parent=None, **kargs):
        super(ImageDraw, self).__init__()
        # self.image=None
        # self.viewbox=viewbox
        self.levels = np.array([0, 255])
        self.lut = None
        self.autoDownsample = False
        self.axisOrder = "row-major"
        self.removable = False

        self.parent = parent
        # kernel[1,1] = 1
        self.setDrawKernel(kernel_size=self.parent.brush_size)
        self.parent.current_stroke = []
        self.parent.in_stroke = False

    def mouseClickEvent(self, ev):
        if self.parent.masksOn:
            if (
                ev.button() == QtCore.Qt.RightButton
                and self.parent.loaded
                and self.parent.nmasks < 2
            ):
                if not self.parent.in_stroke:
                    ev.accept()
                    self.parent.in_stroke = True
                    self.create_start(ev.pos())
                    self.parent.stroke_appended = False
                    self.drawAt(ev.pos(), ev)
                else:
                    ev.accept()
                    self.end_stroke()
                    self.parent.in_stroke = False
            else:
                ev.ignore()
                return
        else:
            ev.ignore()
            return

    def mouseDragEvent(self, ev):
        ev.ignore()
        return

    def hoverEvent(self, ev):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CrossCursor)
        if self.parent.in_stroke:
            if self.parent.in_stroke:
                # continue stroke if not at start
                self.drawAt(ev.pos())
                if self.is_at_start(ev.pos()):
                    self.end_stroke()
                    self.parent.in_stroke = False
        else:
            ev.acceptClicks(QtCore.Qt.RightButton)
            # ev.acceptClicks(QtCore.Qt.LeftButton)

    def create_start(self, pos):
        self.scatter = pg.ScatterPlotItem(
            [pos.x()],
            [pos.y()],
            pxMode=False,
            pen=pg.mkPen(color=(255, 0, 0), width=self.parent.brush_size),
            size=max(3 * 2, self.parent.brush_size * 1.8 * 2),
            brush=None,
        )
        self.parent.p0.addItem(self.scatter)

    def is_at_start(self, pos):
        thresh_out = max(6, self.parent.brush_size * 3)
        thresh_in = max(3, self.parent.brush_size * 1.8)
        # first check if you ever left the start
        if len(self.parent.current_stroke) > 3:
            stroke = np.array(self.parent.current_stroke)
            dist = (
                ((stroke[1:] - stroke[:1][np.newaxis, :, :]) ** 2).sum(axis=-1)
            ) ** 0.5
            dist = dist.flatten()
            # print(dist)
            has_left = (dist > thresh_out).nonzero()[0]
            if len(has_left) > 0:
                first_left = np.sort(has_left)[0]
                has_returned = (dist[max(4, first_left + 1) :] < thresh_in).sum()
                if has_returned > 0:
                    return True
                else:
                    return False
            else:
                return False

    def end_stroke(self):
        self.parent.p0.removeItem(self.scatter)
        if not self.parent.stroke_appended:
            self.parent.stroke = np.array(self.parent.current_stroke)
            self.parent.current_stroke = []
            self.parent.stroke_appended = True
            ioutline = self.parent.stroke[:, -1] == 1
            self.parent.point_set = list(self.parent.stroke[ioutline])
            self.parent.add_set()

    def tabletEvent(self, ev):
        pass
        # print(ev.device())
        # print(ev.pointerType())
        # print(ev.pressure())

    def drawAt(self, pos, ev=None):
        mask = self.greenmask
        set = self.parent.current_point_set
        stroke = self.parent.current_stroke
        pos = [int(pos.y()), int(pos.x())]
        dk = self.drawKernel
        kc = self.drawKernelCenter
        sx = [0, dk.shape[0]]
        sy = [0, dk.shape[1]]
        tx = [pos[0] - kc[0], pos[0] - kc[0] + dk.shape[0]]
        ty = [pos[1] - kc[1], pos[1] - kc[1] + dk.shape[1]]
        kcent = kc.copy()
        if tx[0] <= 0:
            sx[0] = 0
            sx[1] = kc[0] + 1
            tx = sx
            kcent[0] = 0
        if ty[0] <= 0:
            sy[0] = 0
            sy[1] = kc[1] + 1
            ty = sy
            kcent[1] = 0
        if tx[1] >= self.parent.Ly - 1:
            sx[0] = dk.shape[0] - kc[0] - 1
            sx[1] = dk.shape[0]
            tx[0] = self.parent.Ly - kc[0] - 1
            tx[1] = self.parent.Ly
            kcent[0] = tx[1] - tx[0] - 1
        if ty[1] >= self.parent.Lx - 1:
            sy[0] = dk.shape[1] - kc[1] - 1
            sy[1] = dk.shape[1]
            ty[0] = self.parent.Lx - kc[1] - 1
            ty[1] = self.parent.Lx
            kcent[1] = ty[1] - ty[0] - 1

        ts = (slice(tx[0], tx[1]), slice(ty[0], ty[1]))
        ss = (slice(sx[0], sx[1]), slice(sy[0], sy[1]))
        self.image[ts] = mask[ss]

        for ky, y in enumerate(np.arange(ty[0], ty[1], 1, int)):
            for kx, x in enumerate(np.arange(tx[0], tx[1], 1, int)):
                iscent = np.logical_and(kx == kcent[0], ky == kcent[1])
                stroke.append([x, y, iscent])
        self.updateImage()

    def setDrawKernel(self, kernel_size=3):
        bs = kernel_size
        kernel = np.ones((bs, bs), np.uint8)
        self.drawKernel = kernel
        self.drawKernelCenter = [
            int(np.floor(kernel.shape[0] / 2)),
            int(np.floor(kernel.shape[1] / 2)),
        ]
        onmask = 255 * kernel[:, :, np.newaxis]
        offmask = np.zeros((bs, bs, 1))
        opamask = 100 * kernel[:, :, np.newaxis]
        self.redmask = np.concatenate((onmask, offmask, offmask, onmask), axis=-1)
        self.greenmask = np.concatenate((onmask, offmask, onmask, opamask), axis=-1)


class RangeSlider(QSlider):
    """A slider for ranges.

    This class provides a dual-slider for ranges, where there is a defined
    maximum and minimum, as is a normal slider, but instead of having a
    single slider value, there are 2 slider values.

    This class emits the same signals as the QSlider base class, with the
    exception of valueChanged

    Found this slider here: https://www.mail-archive.com/pyqt@riverbankcomputing.com/msg22889.html
    and modified it
    """

    def __init__(self, parent=None, *args):
        super(RangeSlider, self).__init__(*args)

        self._low = self.minimum()
        self._high = self.maximum()

        self.pressed_control = QStyle.SC_None
        self.hover_control = QStyle.SC_None
        self.click_offset = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(
            "QSlider::handle:horizontal {\
                background-color: white;\
                border: 1px solid #5c5c5c;\
                border-radius: 0px;\
                border-color: black;\
                height: 8px;\
                width: 6px;\
                margin: -8px 2; \
                }"
        )

        # self.opt = QtGui.QStyleOptionSlider()
        # self.opt.orientation=QtCore.Qt.Vertical
        # self.initStyleOption(self.opt)
        # 0 for the low, 1 for the high, -1 for both
        self.active_slider = 0
        self.parent = parent

    def level_change(self):
        if self.parent is not None:
            if self.parent.loaded:
                self.parent.saturation = [self._low, self._high]
                self.parent.update_plot()

    def low(self):
        return self._low

    def setLow(self, low):
        self._low = low
        self.update()

    def high(self):
        return self._high

    def setHigh(self, high):
        self._high = high
        self.update()

    def paintEvent(self, event):
        # based on http://qt.gitorious.org/qt/qt/blobs/master/src/gui/widgets/qslider.cpp
        painter = QtGui.QPainter(self)
        style = QtGui.QApplication.style()

        for i, value in enumerate([self._low, self._high]):
            opt = QStyleOptionSlider()
            self.initStyleOption(opt)

            # Only draw the groove for the first slider so it doesn't get drawn
            # on top of the existing ones every time
            if i == 0:
                opt.subControls = (
                    QtGui.QStyle.SC_SliderHandle
                )  # QtGui.QStyle.SC_SliderGroove | QtGui.QStyle.SC_SliderHandle
            else:
                opt.subControls = QtGui.QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                opt.subControls |= QtGui.QStyle.SC_SliderTickmarks

            if self.pressed_control:
                opt.activeSubControls = self.pressed_control
                opt.state |= QtGui.QStyle.State_Sunken
            else:
                opt.activeSubControls = self.hover_control

            opt.sliderPosition = value
            opt.sliderValue = value
            style.drawComplexControl(QtGui.QStyle.CC_Slider, opt, painter, self)

    def mousePressEvent(self, event):
        event.accept()

        style = QtGui.QApplication.style()
        button = event.button()
        # In a normal slider control, when the user clicks on a point in the
        # slider's total range, but not on the slider part of the control the
        # control would jump the slider value to where the user clicked.
        # For this control, clicks which are not direct hits will slide both
        # slider parts
        if button:
            opt = QtGui.QStyleOptionSlider()
            self.initStyleOption(opt)

            self.active_slider = -1

            for i, value in enumerate([self._low, self._high]):
                opt.sliderPosition = value
                hit = style.hitTestComplexControl(
                    style.CC_Slider, opt, event.pos(), self
                )
                if hit == style.SC_SliderHandle:
                    self.active_slider = i
                    self.pressed_control = hit

                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)

                    break

            if self.active_slider < 0:
                self.pressed_control = QtGui.QStyle.SC_SliderHandle
                self.click_offset = self.__pixelPosToRangeValue(
                    self.__pick(event.pos())
                )
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event):
        if self.pressed_control != QtGui.QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_pos = self.__pixelPosToRangeValue(self.__pick(event.pos()))
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)

        if self.active_slider < 0:
            offset = new_pos - self.click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self.active_slider == 0:
            if new_pos >= self._high:
                new_pos = self._high - 1
            self._low = new_pos
        else:
            if new_pos <= self._low:
                new_pos = self._low + 1
            self._high = new_pos

        self.click_offset = new_pos
        self.update()

    def mouseReleaseEvent(self, event):
        self.level_change()

    def __pick(self, pt):
        if self.orientation() == QtCore.Qt.Horizontal:
            return pt.x()
        else:
            return pt.y()

    def __pixelPosToRangeValue(self, pos):
        opt = QtGui.QStyleOptionSlider()
        self.initStyleOption(opt)
        style = QtGui.QApplication.style()

        gr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderGroove, self)
        sr = style.subControlRect(style.CC_Slider, opt, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = sr.width()
            slider_min = gr.x()
            slider_max = gr.right() - slider_length + 1
        else:
            slider_length = sr.height()
            slider_min = gr.y()
            slider_max = gr.bottom() - slider_length + 1

        return style.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            pos - slider_min,
            slider_max - slider_min,
            opt.upsideDown,
        )
