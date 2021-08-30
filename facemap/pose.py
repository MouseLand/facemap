import numpy as np
from PyQt5 import QtGui, QtCore, QtWidgets
import pyqtgraph as pg
from . import transforms, roi
import torch
from . import UNet_helper_functions as utils
import cv2
import time
from PyQt5 import QtGui
import pandas as pd

class Pose():
    def __init__(self, parent):
        self.parent = parent
        self.pose_labels = None
        self.bodyparts = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net = self.load_model()
        self.bbox = self.get_bbox()
        self.img_slice = self.get_img_slice()
        # plot key points  on all frames: predict, save, plot!
        #_  = self.predict_landmarks()

    def get_img_slice(self):
        x1, x2, y1, y2 = self.bbox
        slc = [slice(y1, y2+1) , slice(x1, x2+1)]   # slice img_y and img_x
        print("slice", slc)
        return slc

    def predict_landmarks(self):
        """
        Predict labels for all frames in video and save output as .h5 file
        """
        print("Generating predicted labels using cropped images")

        scorer = "Facemap" # Edit this line and set to user's name
        bodyparts = np.arange(self.net.nout) #temporary, switch to self.net.labels_id
        nchannels = 1
        batch_size = 1

        # Create an empty dataframe
        for index, bodypart in enumerate(bodyparts):
            columnindex = pd.MultiIndex.from_product(
                [[scorer], [bodypart], ["x", "y"]], 
                names=["scorer", "bodyparts", "coords"])
            frame = pd.DataFrame(
                                np.nan,
                                columns=columnindex,
                                index=np.arange(self.parent.cumframes[-1]))
            if index == 0:
                dataFrame = frame
            else:
                dataFrame = pd.concat([dataFrame, frame], axis=1)

        padded = False
        # Get padding details
        if  (self.parent.Ly[0]%16!=0 or self.parent.Lx[0]%16!=0):    # pad image if not divisible by 16 
            padded = True
            im, ysub, xsub = transforms.pad_image_ND(im) # padd random image to get dim
            # slices from padding
            slc = [slice(0, im.shape[n]+1) for n in range(im.ndim)]
            slc[-3] = slice(0, len(bodyparts))
            slc[-2] = slice(ysub[0], ysub[-1]+1)
            slc[-1] = slice(xsub[0], xsub[-1]+1)
        # Store predictions in dataframe
        self.net.eval()
        start = 0
        end = batch_size
        while end != self.parent.cumframes[-1]:
            # Prepocess images
            im = np.zeros((batch_size, nchannels, self.parent.Ly[0], self.parent.Lx[0]))
            for i, frame_ind in enumerate(np.arange(start,end)):
                frame = self.parent.get_frame(frame_ind)[0]
                frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
                im[i] = frame_grayscale_preprocessed
            
            print(start==0)
            # Network prediction 
            hm_pred = self.net(torch.tensor(im).to(device=self.net.DEVICE, dtype=torch.float32)) # convert to tensor and send to device
            if padded:
                hm_pred = hm_pred[slc]       # Slice out padding
            landmarks = utils.heatmap2landmarks(hm_pred.cpu().detach().numpy()).ravel()
            print("lm shape", landmarks.shape)
            dataFrame.iloc[start:end] = landmarks
            print("dataFrame shape", dataFrame.shape)
            start = end 
            end += batch_size
            #pred_df.to_hdf(os.path.join(unet_dir, f.split("/")[-1]+"_"+model_name+".h5"), "df_with_missing", format="table", mode="w")
        return dataFrame

    def save_prediction(self):
        # save .h5 file
        return ""

    def plot_landmarks(self):
        # load labels file
        # Set labels checkbox
        # Plot labels
        """
        self.keypoints_labels = [all_labels[i] for i in sorted(np.unique(all_labels, return_index=True)[1])]
        # Choose colors for each label: provide option for color blindness as well
        self.colors = cm.get_cmap('gist_rainbow')(np.linspace(0, 1., len(self.keypoints_labels)))
        self.colors *= 255
        self.colors = self.colors.astype(int)
        self.colors[:,-1] = 127
        self.brushes = np.array([pg.mkBrush(color=c) for c in self.colors])
        self.parent.Pose_scatterplot.setData(x, y, size=15, symbol='o', brush=self.brushes[filtered_keypoints],
                            hoverable=True, hoverSize=15)
        """
        return ""

    def get_bbox(self):
        bbox_set = False
        prev_bbox = (np.nan, np.nan, np.nan, np.nan)
        while not bbox_set:
            bbox = self.estimate_bbox_region(prev_bbox)
            prev_bbox = bbox
            # plot bbox as ROI
            x1, x2, y1, y2 = bbox
            dx, dy = x2-x1, y2-y1
            self.bbox_plot = roi.sROI(rind=1, rtype="bbox", iROI=1, moveable=False, 
                                    parent=self.parent, pos=(y1, x1, dy, dx))
            # get user validation
            qm = QtGui.QMessageBox
            ret = qm.question(self.parent,'', "Does the suggested ROI match the requirements?", qm.Yes | qm.No)
            bbox_set = ret == qm.Yes
            if not bbox_set:
                del self.bbox_plot
        print("bbox set!")
        bbox = np.round(bbox).astype(int)
        return bbox

    def estimate_bbox_region(self, prev_bbox):
        """
        Obtain ROI/bbox for cropping images to use as input for model 
        """
        num_iter = 3 # select optimum value
        for it in range(num_iter):
            t0 = time.time()
            imgs = self.get_batch_imgs()
            # Get bounding box regions
            bbox, _ = transforms.get_bounding_box(imgs, self.net, prev_bbox)
            prev_bbox = bbox    # Update bbox 
            adjusted_bbox = transforms.adjust_bbox(bbox, (self.parent.Ly[0], self.parent.Lx[0])) # Adjust bbox to be square instead of rectangle
            print(adjusted_bbox, "bbox", time.time()-t0) 
        return adjusted_bbox
        
    def load_model(self):
        """
        Load pre-trained UNet model for labels prediction 
        """
        model_file = "/Users/Atika/Neuroverse/Janelia/facemap-mac/facemap/example_model_small"  # Replace w/ function that downloads model from server
        self.bodyparts = None #data['landmarks']
        kernel_size = 3
        nbase = [1, 64, 64*2, 64*3, 64*4] # number of channels per layer
        nout = 14 #len(self.bodyparts)  # number of outputs
        net = utils.UNet_b(nbase, nout, kernel_size, labels_id=self.bodyparts)
        if torch.cuda.is_available():
            cpu_is_device = False
        else:
            cpu_is_device = True
        net.load_model(model_file, cpu=cpu_is_device)
        return net

    def get_batch_imgs(self, batch_size=1, nchannels=1):
        """
        Get batch of images sampled randomly from video. Note: works for single video only
        Parameters
        -------------
        self: (Pose) object
        batch_size: int number
        Returns
        --------------
        im: 1-D list
            list of images each of size (Ly x Lx) 
        """
        img_ind = np.random.randint(0, self.parent.cumframes[-1], batch_size)
        im = np.zeros((batch_size, nchannels, self.parent.Ly[0], self.parent.Lx[0]))
        for k, ind in enumerate(img_ind):
            frame = self.parent.get_frame(ind)[0]
            frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_grayscale_preprocessed = transforms.preprocess_img(frame_grayscale)
            im[k] = frame_grayscale_preprocessed
        return im
