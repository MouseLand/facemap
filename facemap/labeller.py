import sys
import os
import shutil
import time
import numpy as np
from PyQt5 import QtGui, QtCore, Qt, QtWidgets
import pyqtgraph as pg
from pyqtgraph import GraphicsScene
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from skimage import io
from skimage import transform, draw, measure, segmentation
import warnings
from .gui import guiparts
from guiparts import ImageDraw, RangeSlider, RGBRadioButtons, ViewBoxNoRightDrag
import matplotlib.pyplot as plt
import copy
import mxnet as mx
from mxnet import nd
from glob import glob
from natsort import natsorted
import argparse

def make_bwr():
    # make a bwr colormap
    b = np.append(255*np.ones(128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
    r = np.append(np.linspace(0, 255, 128), 255*np.ones(128))[:,np.newaxis]
    g = np.append(np.linspace(0, 255, 128), np.linspace(0, 255, 128)[::-1])[:,np.newaxis]
    color = np.concatenate((r,g,b), axis=-1).astype(np.uint8)
    bwr = pg.ColorMap(pos=np.linspace(0.0,255,256), color=color)
    return bwr

def get_unique_points(set):
    cps = np.zeros((len(set),3), np.int32)
    for k,pp in enumerate(set):
        cZ, posy, posx = pp[0], pp[1], pp[2]
        cps[k,:] = np.array(pp)
    set = list(np.unique(cps, axis=0))
    return set

class MainW(QtGui.QMainWindow):
    def __init__(self, images=None):
        super(MainW, self).__init__()

        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(50, 50, 850, 850)
        self.setWindowTitle("train eyenet")
        icon_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "logo/logo.png"
        )
        app_icon = QtGui.QIcon()
        app_icon.addFile(icon_path, QtCore.QSize(16, 16))
        app_icon.addFile(icon_path, QtCore.QSize(24, 24))
        app_icon.addFile(icon_path, QtCore.QSize(32, 32))
        app_icon.addFile(icon_path, QtCore.QSize(48, 48))
        app_icon.addFile(icon_path, QtCore.QSize(96, 96))
        app_icon.addFile(icon_path, QtCore.QSize(256, 256))


        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("&File")
        # load processed data
        loadImg = QtGui.QAction("&Load image (*.tif, *.png, *.jpg)", self)
        loadImg.setShortcut("Ctrl+L")
        loadImg.triggered.connect(lambda: self.load_images(images))
        file_menu.addAction(loadImg)

        #self.setWindowIcon(app_icon)
        self.setStyleSheet("QMainWindow {background: 'black';}")
        self.stylePressed = ("QPushButton {Text-align: left; "
                             "background-color: rgb(100,50,100); "
                             "color:white;}")
        self.styleUnpressed = ("QPushButton {Text-align: left; "
                               "background-color: rgb(50,50,50); "
                               "color:white;}")
        self.styleInactive = ("QPushButton {Text-align: left; "
                              "background-color: rgb(50,50,50); "
                              "color:gray;}")
        self.loaded = False

        # ---- MAIN WIDGET LAYOUT ---- #
        self.cwidget = QtGui.QWidget(self)
        self.l0 = QtGui.QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.setCentralWidget(self.cwidget)
        self.l0.setVerticalSpacing(4)

        self.imask = 0

        # ---- drawing area ---- #
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 2,3, 12, 10)
        layout = self.win.ci.layout
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        self.win.scene().sigMouseMoved.connect(self.mouse_moved)
        self.make_viewbox()
        bwrmap = make_bwr()
        self.bwr = bwrmap.getLookupTable(start=0.0, stop=255.0, alpha=False)


        self.make_buttons()

        self.colormap = (plt.get_cmap('gist_rainbow')(np.linspace(0.0,1.0,1000)) * 255).astype(np.uint8)
        self.reset()

        if images is not None:
            self.filename = images
            self.load_images(self.filename)

        self.setAcceptDrops(True)

        print(self.loaded)
        self.win.show()
        self.show()

    def make_buttons(self):
        self.slider = RangeSlider(self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(255)
        self.slider.setLow(0)
        self.slider.setHigh(255)
        self.slider.setTickPosition(QtGui.QSlider.TicksBelow)
        self.l0.addWidget(self.slider, 3,0,1,1)

        self.brush_size = 3
        self.BrushChoose = QtGui.QComboBox()
        self.BrushChoose.addItems(["1","3","5","7"])
        self.BrushChoose.currentIndexChanged.connect(self.brush_choose)
        self.l0.addWidget(self.BrushChoose, 1, 5,1,1)
        label = QtGui.QLabel('brush size:')
        label.setStyleSheet('color: white;')
        self.l0.addWidget(label, 0, 5,1,1)

        # cross-hair
        self.vLine = pg.InfiniteLine(angle=90, movable=False)
        self.hLine = pg.InfiniteLine(angle=0, movable=False)

        # turn on crosshairs
        self.CHCheckBox = QtGui.QCheckBox('cross-hairs')
        self.CHCheckBox.setStyleSheet('color: white;')
        self.CHCheckBox.toggled.connect(self.cross_hairs)
        self.l0.addWidget(self.CHCheckBox, 1,4,1,1)

        # turn off masks
        self.MCheckBox = QtGui.QCheckBox('masks on [SPACE]')
        self.MCheckBox.setStyleSheet('color: white;')
        self.MCheckBox.setChecked(True)
        self.MCheckBox.toggled.connect(self.masks_on)
        self.l0.addWidget(self.MCheckBox, 0,6,1,1)
        self.masksOn=True

        # clear all masks
        self.ClearButton = QtGui.QPushButton('clear all masks')
        self.ClearButton.clicked.connect(self.clear_all)
        self.l0.addWidget(self.ClearButton, 1,6,1,1)
        self.ClearButton.setEnabled(False)

        # choose models
        self.ModelChoose = QtGui.QComboBox()
        self.model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
        models = glob(self.model_dir+'/*')
        models = [os.path.split(m)[-1] for m in models]
        print(models)
        self.ModelChoose.addItems(models)
        self.l0.addWidget(self.ModelChoose, 1, 7,1,1)
        label = QtGui.QLabel('model: ')
        label.setStyleSheet('color: white;')
        self.l0.addWidget(label, 0, 7,1,1)

        # recompute model
        self.ModelButton = QtGui.QPushButton('compute masks')
        self.ModelButton.clicked.connect(self.compute_model)
        self.l0.addWidget(self.ModelButton, 1,10,1,1)
        self.ModelButton.setEnabled(False)
        self.imgLR = False
        self.saturation = [0,255]

    def keyPressEvent(self, event):
        if self.loaded:
            #self.p0.setMouseEnabled(x=True, y=True)
            if (event.modifiers() != QtCore.Qt.ControlModifier and
                event.modifiers() != QtCore.Qt.ShiftModifier and
                event.modifiers() != QtCore.Qt.AltModifier):
                if not self.in_stroke:
                    if len(self.current_point_set) > 0:
                        if event.key() == QtCore.Qt.Key_Return:
                            self.add_set()
                    else:
                        if event.key() == QtCore.Qt.Key_Space:
                            self.MCheckBox.toggle()
                    if event.key() == QtCore.Qt.Key_Left:
                        self.get_prev_image()
                    elif event.key() == QtCore.Qt.Key_Right:
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_A:
                        self.get_prev_image()
                    elif event.key() == QtCore.Qt.Key_D:
                        self.get_next_image()
                    elif event.key() == QtCore.Qt.Key_Up:
                        self.BrushChoose.setCurrentIndex(max(self.BrushChoose.currentIndex()-1, 0))
                        self.brush_choose()
                    elif event.key() == QtCore.Qt.Key_Down:
                        self.BrushChoose.setCurrentIndex(min(self.BrushChoose.currentIndex()+1, 3))
                        self.brush_choose()
                self.update_plot()
            elif event.modifiers() == QtCore.Qt.ControlModifier:
                if event.key() == QtCore.Qt.Key_Z:
                    if self.nmasks > 0:
                        self.remove_mask()
                        self.save_sets()

    def get_files(self):
        images = []
        images.extend(glob(os.path.dirname(self.filename) + '/*.png'))
        images.extend(glob(os.path.dirname(self.filename) + '/*.jpg'))
        images.extend(glob(os.path.dirname(self.filename) + '/*.jpeg'))
        images.extend(glob(os.path.dirname(self.filename) + '/*.tif'))
        images.extend(glob(os.path.dirname(self.filename) + '/*.tiff'))
        images = sorted(images)

        fnames = [os.path.split(images[k])[-1] for k in range(len(images))]
        f0 = os.path.split(self.filename)[-1]

        idx = np.nonzero(np.array(fnames)==f0)[0][0]

        #idx = np.nonzero(np.array(images)==self.filename[0])[0][0]
        return images, idx

    def get_prev_image(self):
        images, idx = self.get_files()
        idx = (idx-1)%len(images)
        #print(images[idx-1])
        self.imgLR = True
        self.load_images(filename=images[idx])

    def get_next_image(self):
        images, idx = self.get_files()
        idx = (idx+1)%len(images)
        #print(images[idx+1])
        self.imgLR = True
        self.load_images(filename=images[idx])

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        print(files)
        self.load_images(filename=files[0])

    def masks_on(self):
        self.masksOn = (self.masksOn+1)%2
        if self.masksOn:
            self.p0.addItem(self.layer)
        else:
            self.p0.removeItem(self.layer)
        if self.loaded:
            self.update_plot()
        self.update_plot()

    def make_viewbox(self):
        self.p0 = ViewBoxNoRightDrag(
            lockAspect=True,
            name="plot1",
            border=[100, 100, 100],
            invertY=True
        )
        self.brush_size=3
        self.win.addItem(self.p0, 0, 0)
        self.p0.setMenuEnabled(False)
        self.p0.setMouseEnabled(x=True, y=True)
        self.img = pg.ImageItem(viewbox=self.p0, parent=self)
        self.img.autoDownsample = False
        self.layer = ImageDraw(viewbox=self.p0, parent=self)
        self.layer.setLevels([0,255])
        self.p0.scene().contextMenuItem = self.p0
        #self.p0.setMouseEnabled(x=False,y=False)
        self.Ly,self.Lx = 512,512
        self.p0.addItem(self.img)
        self.p0.addItem(self.layer)

    def reset(self):
        # ---- start sets of points ---- #
        self.loaded = False
        self.current_point_set = []
        self.masks = []
        self.outlines = []
        self.nmasks = 0
        # -- set menus to default -- #
        #self.BrushChoose.setCurrentIndex(1)
        self.CHCheckBox.setChecked(False)

        # -- zero out image stack -- #
        self.opacity = 200 # how opaque masks should be
        self.colors = np.array([np.array([255,255,255,0]),
                        np.array([0,0,255, self.opacity]),
                        np.array([255,0,0, self.opacity])])
        self.Ly, self.Lx = 512,512
        self.currentZ = 0
        self.stack = np.zeros((self.Ly,self.Lx))
        self.layers = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.maskpix = -1*np.ones((self.Ly,self.Lx), np.int32)
        self.outpix = -1*np.ones((self.Ly,self.Lx), np.int32)
        self.update_plot()
        self.basename = []
        self.filename = []
        self.loaded = False

    def brush_choose(self):
        self.brush_size = self.BrushChoose.currentIndex()*2 + 1
        if self.loaded:
            self.layer.setDrawKernel(kernel_size=self.brush_size)
            self.update_plot()

    def cross_hairs(self):
        if self.CHCheckBox.isChecked():
            self.p0.addItem(self.vLine, ignoreBounds=True)
            self.p0.addItem(self.hLine, ignoreBounds=True)
        else:
            self.p0.removeItem(self.vLine)
            self.p0.removeItem(self.hLine)

    def clear_all(self):
        self.layers = 0*np.ones((self.Ly,self.Lx,4), np.uint8)
        self.maskpix = 0*np.ones((self.Ly,self.Lx), np.int32)
        self.nmasks = 0
        self.masks = []
        self.outlines = []
        print('removed masks')
        self.ClearButton.setEnabled(False)
        self.update_plot()

    def plot_clicked(self, event):
        if event.double():
            if event.button()==QtCore.Qt.LeftButton:
                if (event.modifiers() != QtCore.Qt.ShiftModifier and
                    event.modifiers() != QtCore.Qt.AltModifier):
                    self.p0.setYRange(0,self.Ly)
                    self.p0.setXRange(0,self.Lx)

    def mouse_moved(self, pos):
        items = self.win.scene().items(pos)
        for x in items:
            if x==self.p0:
                mousePoint = self.p0.mapSceneToView(pos)
                if self.CHCheckBox.isChecked():
                    self.vLine.setPos(mousePoint.x())
                    self.hLine.setPos(mousePoint.y())

    def update_plot(self):
        self.Ly, self.Lx = self.stack.shape
        image = self.stack
        self.img.setImage(image, autoLevels=False, lut=None)
        self.img.setLevels(self.saturation)

        if self.masksOn:
            self.layer.setImage(self.layers, autoLevels=False)
        self.slider.setLow(self.saturation[0])
        self.slider.setHigh(self.saturation[1])
        self.win.show()
        self.show()

    def remove_stroke(self, delete_points=True):
        #self.current_stroke = get_unique_points(self.current_stroke)
        stroke = np.array(self.stroke)
        self.layers = self.colors[self.maskpix]
        if delete_points:
            self.point_set = []
            self.stroke = []
        self.update_plot()


    def remove_mask(self):
        if self.nmasks==2:
            self.maskpix = np.zeros((self.Ly,self.Lx), np.int32)
            self.maskpix[self.outlines[0][:,0], self.outlines[0][:,1]] = 1
            self.layers = self.colors[self.maskpix]
        else:
            self.maskpix = np.zeros((self.Ly,self.Lx), np.int32)
            self.layers = np.zeros((self.Ly,self.Lx,4), np.uint8)
        self.nmasks -= 1
        del self.masks[-1]
        del self.outlines[-1]
        print(len(self.masks))
        print('removed 1 mask')
        if self.nmasks==0:
            self.ClearButton.setEnabled(False)
        self.update_plot()


    def add_set(self, pts=None, save=True):
        if pts is None and len(self.point_set) > 0:
            self.point_set = np.array(self.point_set)
            self.remove_stroke(delete_points=False)
            pts = self.outline_to_mask(self.point_set)
        if pts is not None:
            outline, pts = pts
            self.maskpix[outline[:,0], outline[:,1]] = int(self.nmasks+1)
            self.layers = self.colors[self.maskpix]
            self.masks.append(pts)
            self.outlines.append(outline)
            self.ClearButton.setEnabled(True)
            self.nmasks+=1
            if save:
                self.save_sets()
        self.stroke = []
        self.point_set = []
        self.update_plot()


    def outline_to_mask(self, points):
        if 1:
            vr = points[:,0]
            vc = points[:,1]
            vr, vc = draw.polygon_perimeter(vr, vc, self.layers.shape)
            ar, ac = draw.polygon(vr, vc, self.layers.shape)
            ar = np.append(ar, vr)
            ac = np.append(ac, vc)
            pts = np.hstack((ar[:,np.newaxis], ac[:,np.newaxis]))
            pts = np.unique(pts, axis=0)
            outline = np.hstack((vr[:,np.newaxis], vc[:,np.newaxis]))
            if pts.shape[0] < 5:
                print('cell too small')
                return None
            else:
                return outline, pts
        else:
            print('ERROR: not a shape')
            return None

    def save_sets(self):
        base = os.path.splitext(self.filename)[0]
        np.save(base + '_manual.npy',
                    {'img': self.stack,
                     'masks': self.masks,
                     'outlines': self.outlines,
                     'filename': self.filename})
        #print(self.point_sets)
        print('--- %d masks saved'%(len(self.masks)))

    def initialize_images(self, image):
        self.onechan=False
        if image.ndim==3:
            if image.shape[0] < 5:
                image = np.transpose(image, (1,2,0))

            if image.shape[-1] < 3:
                shape = image.shape
                image = np.concatenate((image,
                                       np.zeros((shape[0], shape[1], 3-shape[2]),
                                        dtype=type(image[0,0,0]))), axis=-1)
                if 3-shape[2]>1:
                    self.onechan=True
            elif image.shape[-1]<5 and image.shape[-1]>2:
                image = image[:,:,:3]
            image = image.astype(np.float32).mean(axis=-1)
        else:
            image = image.astype(np.float32)
        print(image.shape)

        self.stack = image
        if self.stack.max()>255 or self.stack.min()<0:
            self.stack -= self.stack.min()
            self.stack /= self.stack.max()
            self.stack *= 255
        self.layers = 255*np.ones((image.shape[0], image.shape[1], 4), dtype=np.uint8)
        self.layers[:,:,-1] = 0 # set transparent
        self.maskpix = np.zeros(image.shape[:2], np.int32)
        self.medians = []
        if not self.imgLR:
            self.compute_saturation()

    def load_manual(self, filename=None, image=None, image_file=None):
        if filename is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Load manual labels", filter="*_manual.npy"
                )
            filename = name[0]
        try:
            dat = np.load(filename, allow_pickle=True).item()
            dat['masks']
            self.loaded = True
        except:
            self.loaded = False
            print('not NPY')
            return

        self.reset()
        if image is None:
            if 'filename' in dat:
                self.filename = dat['filename']
                if image is None:
                    if os.path.isfile(self.filename):
                        self.filename = dat['filename']
                    else:
                        imgname = os.path.split(self.filename)[1]
                        root = os.path.split(filename)[0]
                        self.filename = root+'/'+imgname
                try:
                    image = io.imread(self.filename)
                except:
                    self.loaded = False
                    print('ERROR: cannot find image')
                    return
            else:
                self.filename = filename[:-11]
                if image is None:
                    image = dat['img']
        else:
            self.filename = image_file
        print(self.filename)

        self.initialize_images(image)
        self.masks = dat['masks']
        self.outlines = dat['outlines']
        for n in range(len(self.masks)):
            self.add_set(pts=[self.outlines[n], self.masks[n]], save=False)
        self.loaded = True
        print('%d masks found'%(self.nmasks))
        self.enable_buttons()
        self.update_plot()

    def compute_saturation(self):
        # compute percentiles from stack
        self.saturation = [np.percentile(self.stack.astype(np.float32),1),
                           np.percentile(self.stack.astype(np.float32),99)]

    def compute_model(self):
        self.clear_all()
        self.flows = [[],[],[],[]]
        device = mx.cpu()
        if not hasattr(self, 'net'):
            self.net = unet.Net2D()
            self.net.hybridize()
            self.net.initialize(ctx = device)
            self.current_model = self.ModelChoose.currentText()
            print(self.current_model)
            self.net.load_parameters(os.path.join(self.model_dir, self.current_model))
        elif self.ModelChoose.currentText() != self.current_model:
            self.net = unet.Net2D()
            self.net.hybridize()
            self.net.initialize(ctx = device)
            self.current_model = self.ModelChoose.currentText()
            print(self.current_model)
            self.net.load_parameters(os.path.join(self.model_dir, self.current_model))

        for z in range(len(self.stack)):
            image = self.stack[z].astype(np.float32)
            # use grayscale image
            image = self.chanchoose(image)
            # rescale image
            try:
                x = utils.normalize99(image)
                Ly,Lx = x.shape[-2:]
                imgi, pads = transforms.pad_image_CS0(x)
                while imgi.ndim<4:
                    imgi = np.expand_dims(imgi, 0)
                X = nd.array(imgi, ctx=device)
                # run network
                y = self.net(X)[0]
                print(X.shape)
            except:
                # add zero channel
                if image.shape[0]==1:
                    image = np.concatenate((image, np.zeros_like(image)), axis=0)
                else:
                    image = image[:1]
                x = utils.normalize99(image)
                Ly,Lx = x.shape[-2:]
                imgi, pads = transforms.pad_image_CS0(x)
                while imgi.ndim<4:
                    imgi = np.expand_dims(imgi, 0)
                X = nd.array(imgi, ctx=device)
                # run network
                y = self.net(X)[0]
                print(X.shape)
            y = y.detach().asnumpy()
            # undo padding
            y = y[:, :, pads[-2][0]:y.shape[-2]-pads[-2][-1], pads[-1][0]:y.shape[-1]-pads[-1][-1]]
            print(y.shape)
            # compute dynamics from flows
            incell = y[0,2] > .0
            yout = utils.run_dynamics(-y[:,:2] * incell)
            masks = utils.get_mask(yout[0])[0]
            y[0,0] /= np.abs(y[0,0]).max()
            y[0,1] /= np.abs(y[0,1]).max()
            self.flows[0].append((y[0,0] * 127 + 127).astype(np.uint8))
            self.flows[1].append((y[0,1] * 127 + 127).astype(np.uint8))
            if 0:
                self.flows[2].append((y[0,1] * 127 + 127).astype(np.uint8))
            else:
                self.flows[2].append(np.zeros((Ly,Lx), np.uint8))
            self.flows[-1].append(np.clip(y[0,-1] * 127 + 127, 0, 255).astype(np.uint8))

            yc = np.array([0,Ly-1,0,Ly-1])
            xc = np.array([0,0,Lx-1,Lx-1])

            print('%d cells found with unet'%(len(np.unique(masks)[1:])))

            for n in np.unique(masks)[1:]:
                col_rand = np.random.randint(1000)
                color = self.colormap[col_rand,:3]
                mn = (masks==n).astype(np.uint8)
                nc0 = self.ncells
                if mn.sum() > 5:
                    pix = self.find_outline(mn=mn)
                    if pix is not None:
                        pix = np.concatenate((z*np.ones_like(pix[:,:1]),
                                            pix,
                                            np.ones_like(pix[:,:1])), axis=-1)
                        median = self.add_mask(points=pix, color=color, mask=mn.nonzero())
                        if median is not None:
                            self.cellcolors.append(color)
                            self.ncells+=1
                            for z,m in enumerate(median):
                                self.medians[z].append(m)
            if self.ncells>0:
                self.ClearButton.setEnabled(True)
            print(self.ncells)

    def load_images(self, filename=None):
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        types = ["*.png","*.jpg","*.tif","*.tiff"] # supported image types
        if filename is None:
            name = QtGui.QFileDialog.getOpenFileName(
                self, "Load image"
                )
            filename = name[0]
        manual_file = os.path.splitext(filename)[0]+'_manual.npy'
        if os.path.isfile(manual_file):
            print(manual_file)
            self.load_manual(manual_file, image=io.imread(filename), image_file=filename)
            return
        try:
            image = io.imread(filename)
            self.loaded = True
        except:
            print('images not compatible')

        self.prediction = False
        if self.loaded:
            self.reset()
            self.filename = filename
            print(filename)
            self.basename, filename = os.path.split(self.filename)
            self.initialize_images(image)
            self.loaded = True
            self.enable_buttons()
        QtWidgets.QApplication.restoreOverrideCursor()


    def enable_buttons(self):
        self.ModelButton.setEnabled(True)
        if self.nmasks > 0:
            self.ClearButton.setEnabled(True)
        self.update_plot()
        self.setWindowTitle(self.filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='image files')
    parser.add_argument('--file', default=[], type=str, help='options')
    args = parser.parse_args()

    if len(args.file)>0:
        image = args.file
    else:
        image = None
    # Always start by initializing Qt (only once per application)
    warnings.filterwarnings("ignore")
    app = QtGui.QApplication(sys.argv)
    icon_path = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "logo/logo.png"
    )
    app_icon = QtGui.QIcon()
    app_icon.addFile(icon_path, QtCore.QSize(16, 16))
    app_icon.addFile(icon_path, QtCore.QSize(24, 24))
    app_icon.addFile(icon_path, QtCore.QSize(32, 32))
    app_icon.addFile(icon_path, QtCore.QSize(48, 48))
    app_icon.addFile(icon_path, QtCore.QSize(96, 96))
    app_icon.addFile(icon_path, QtCore.QSize(256, 256))
    app.setWindowIcon(app_icon)

    #zstack = 'x.npy'
    GUI = MainW(images=image)
    ret = app.exec_()
    # GUI.save_gui_data()
    sys.exit(ret)
