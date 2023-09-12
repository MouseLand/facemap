import torch
from torch.utils.data import Dataset
from facemap.pose import transforms
from facemap.pose import pose_helper_functions as pose_utils
from facemap.tongue import segmentation_utils
import torch.nn.functional as F

class TongueDataset(Dataset):
    """Tongue dataset."""

    def __init__(self, img, bbox=None, threshold=4.0, img_size=(256, 256), train=True):
        """
        Args:
            img (ND-array): Image data.
            mask (ND-array): Mask data.
            bbox (ND-array): Bounding box data of the form [[x1, x2, y1, y2]].
            img_size (tuple): Size of the image to be returned.
            train (bool): Whether the dataset is for training or testing.
        """
        self.img = img
        self.bbox = bbox
        self.threshold = threshold
        self.train = train
        self.img_size = img_size
        self.img = self.preprocess_imgs(self.img, bbox=self.bbox)
        
    def __len__(self):
        return len(self.img)

    def __getitem__(self, idx):
        img = self.img[idx]

        if self.train:
            img = segmentation_utils.augment_data(img)


        # If not a tensor, convert to tensor
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)

        sample = {'image': img, 'idx': idx}

        return sample

    def preprocess_imgs(self, image_data, bbox):
        """
        Preprocess images to be in the range [0, 1] and normalize99
        Parameters
        ----------
        image_data : list of ND-array of shape (C, W, H)
            List of images.
        Returns
        -------
        image_data : list of ND-array of shape (C, W, H)
            List of images.
        """
        imgs = []
        for im in image_data:
            im = torch.from_numpy(im)
            # Normalize
            im = pose_utils.normalize99(im)
            # 1. Crop image
            im = transforms.crop_image(im, bbox)
             # 2. Pad image to square
            im, _ = transforms.pad_img_to_square(im)
            # 3. Resize image to resize_shape for model input
            im = resize_image(im, self.img_size)
            imgs.append(im)
        return imgs
    
def resize_image(im, resize_shape):
    """
    Resize image to given height and width.
    Parameters
    ----------
    im : ND-array
        image of size [Ly x Lx]
    resize_shape : tuple of size (2,)
        desired shape of image
    Returns
    -------
    im : ND-array
        resized image of size [h x w]
    """
    h, w = resize_shape
    if im.ndim == 3:
        im = torch.unsqueeze(im, dim=0)
    elif im.ndim == 2:
        im = torch.unsqueeze(im, dim=0)
        im = torch.unsqueeze(im, dim=0)
    im = F.interpolate(im, size=(h, w), mode="bilinear", align_corners=True).squeeze(dim=0)
    return im

