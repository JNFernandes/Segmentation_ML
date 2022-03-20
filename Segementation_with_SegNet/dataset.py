import os
from skimage.io import imread
from torch.utils.data import Dataset
import numpy as np


class SegDataset(Dataset):
    def __init__(self, img_directory, mask_directory, transform=None):
        self.image_dir = img_directory
        self.mask_dir = mask_directory
        self.transform = transform
        self.images = sorted(os.listdir(img_directory))
        self.masks = sorted(os.listdir(mask_directory))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        msk = self.masks[index]
        path_join_image = os.path.join(self.image_dir, img)
        path_join_mask = os.path.join(self.mask_dir, msk)

        # image processing
        image_path = path_join_image
        image = np.array(imread(image_path).astype(np.float32)/255)

        # mask processing
        mask_path = path_join_mask
        mask = np.array(imread(mask_path).astype(np.float32))  # 0.0 for black, 255 .0for white
        mask[mask == 255.0] = 1.0  # sigmoid on our last activation function

        if self.transform is not None:
            augmented_data = self.transform(image=image, mask=mask)
            image = augmented_data["image"]
            mask = augmented_data["mask"]

        return image, mask









