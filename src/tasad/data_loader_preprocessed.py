import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob

class MVTecPreprocessedDataset(Dataset):

    def __init__(self, root_dir, datatype='png', resize_shape=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir + f"*.{datatype}"))
        
        if len(self.images) == 0:
            self.images = sorted(glob.glob(root_dir + f"/*/*.{datatype}"))
            
        self.resize_shape = resize_shape

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))

        if self.resize_shape is not None:
            image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.expand_dims(mask, axis=2).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = False
        else:
            mask_dir_path = os.path.join(dir_path, '../../ground_truth/', base_dir)
            mask_file_name = file_name.split(".")[0] + ".jpg"
            mask_path = os.path.join(mask_dir_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = True

        return {
            # 'image': image, 
            'augmented_image': image, 
            'has_anomaly': has_anomaly,
            'anomaly_mask': mask, 
            'idx': idx
        }