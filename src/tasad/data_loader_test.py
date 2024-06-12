import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from src.tasad.utils.perlin import rand_perlin_2d_np

class MVTecTestDataset(Dataset):

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
            mask_file_name = file_name.split(".")[0] + "_mask.png"
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

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None):
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + "*.png"))
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + "/*/*.png"))

        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
            iaa.Solarize(0.5, threshold=(32, 128)),
            iaa.Posterize(),
            iaa.Invert(),
            iaa.pillike.Autocontrast(),
            iaa.pillike.Equalize(),
            iaa.Affine(rotate=(-45, 45))
        ]

        self.rotator = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])

    def __len__(self):
        return len(self.image_paths)

    def random_augmenter(self):
        augmenter_indices = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        augmenter = iaa.Sequential([self.augmenters[idx] for idx in augmenter_indices])
        return augmenter

    def augment_image(self, image, anomaly_source_path):
        augmenter = self.random_augmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))

        anomaly_img_augmented = augmenter(image=anomaly_source_img)
        
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rotator(image=perlin_noise)
        
        threshold = 0.5
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        perlin_thr = np.expand_dims(perlin_thr, axis=2)

        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        beta = torch.rand(1).numpy()[0] * 0.8

        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * perlin_thr

        if torch.rand(1).numpy()[0] > 0.5:
            return image.astype(np.float32), np.zeros_like(perlin_thr, dtype=np.float32), np.array([0.0], dtype=np.float32)
        else:
            augmented_image = augmented_image.astype(np.float32)
            mask = perlin_thr.astype(np.float32)
            augmented_image = mask * augmented_image + (1 - mask) * image
            has_anomaly = 1.0
            
            if np.sum(mask) == 0:
                has_anomaly = 0.0

            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        if torch.rand(1).numpy()[0] > 0.7:
            image = self.rotator(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(
            self.image_paths[idx], self.anomaly_source_paths[anomaly_source_idx])

        return {
            'image': image, 
            'anomaly_mask': anomaly_mask,
            'augmented_image': augmented_image, 
            'has_anomaly': has_anomaly, 
            'idx': idx
        }
