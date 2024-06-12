import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from src.tasad.utils.utilts_func import *
from src.tasad.utils.perlin import rand_perlin_2d_np
from skimage.segmentation import slic
from skimage.metrics import structural_similarity as ssim

class MVTecTrainDataset(Dataset):
    def __init__(self, root_dir, anomaly_source_path, anomaly_choices=[0,1], resize_shape=None, include_norm_imgs=1, datatype='png'):
        """
        Args:
            root_dir (string): Directory with all the images.
            anomaly_source_path (string): Directory with anomaly images.
            anomaly_choices (list): List of anomaly choices.
            resize_shape (tuple): Shape to resize images.
            include_norm_imgs (int): Flag to include normal images.
            datatype (str): Image file type.
        """
        self.root_dir = root_dir
        self.resize_shape = resize_shape

        self.image_paths = sorted(glob.glob(root_dir + f"*.{datatype}"))
        
        if len(self.image_paths) == 0:
            self.image_paths = sorted(glob.glob(root_dir + f"/*/*.{datatype}"))

        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + '/*/*.png'))
        
        if len(self.anomaly_source_paths) == 0:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + '/*/*.jpg'))
            
        if len(self.anomaly_source_paths) == 0:
            self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path + f'*.{datatype}'))

        if include_norm_imgs != 0:
            self.anomaly_source_paths = add_norm_imgs_to_anom_imgs(self.anomaly_source_paths, self.image_paths)
        
        self.anomaly_type = anomaly_choices
        
        self.augmenters = [
            iaa.GammaContrast((0.5, 2.0), per_channel=True),
            iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
            iaa.pillike.EnhanceSharpness(),
            iaa.pillike.Autocontrast(),
        ]

    def __len__(self):
        return len(self.image_paths)

    def random_augmenter(self):
        augmenter_indices = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        augmenter = iaa.Sequential([self.augmenters[idx] for idx in augmenter_indices])
        return augmenter

    def augment_image(self, image, anomaly_source_path, anomaly_type=[0,1]):
        augmenter = self.random_augmenter()
        
        perlin_scale = 6
        min_perlin_scale = 0
        
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        anomaly_img_augmented = augmenter(image=anomaly_source_img)
        
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

        anomaly_choice = np.random.choice(self.anomaly_type, 1)[0]
        seg_choices = [30, 50, 80]
        seg_choice = np.random.choice(seg_choices, len(seg_choices))[0]
        
        if anomaly_choice == 0:
            rotate_choice = np.random.choice(3, 3)[0]
            if rotate_choice == 0:
                anomaly_img_augmented = cv2.rotate(anomaly_img_augmented, cv2.ROTATE_90_CLOCKWISE)
            elif rotate_choice == 1:
                anomaly_img_augmented = cv2.rotate(anomaly_img_augmented, cv2.ROTATE_180)
            
            segments_slic = slic(anomaly_img_augmented.reshape((256, 256, 3)), n_segments=seg_choice, compactness=8, sigma=1, start_label=1)
            unique_segments = np.unique(segments_slic)
            selected_segments = np.random.choice(unique_segments, int(len(unique_segments) * 0.1))
            segment_mask = np.isin(segments_slic, selected_segments)
            mask = np.expand_dims(segment_mask, axis=2)
            augmented_image = anomaly_img_augmented.astype(np.float32) / 255              
            augmented_image = mask * augmented_image + (1 - mask) * image
        elif anomaly_choice == 1:
            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
            perlin_threshold = np.where(perlin_noise > np.random.rand(1)[0], np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_threshold = np.expand_dims(perlin_threshold, axis=2)
            mask = np.copy(perlin_threshold)
            img_threshold = (anomaly_img_augmented.astype(np.float32) * perlin_threshold) / 255.0
            beta = torch.rand(1).numpy()[0] * 0.8
            augmented_image = image * (1 - perlin_threshold) + (1 - beta) * img_threshold + beta * image * perlin_threshold
        
        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5 or anomaly_type == 2:
            return image, np.expand_dims(np.zeros_like(image[:, :, 0], dtype=np.float32), axis=2), np.array([0.0], dtype=np.float32)
        else:
            try:
                augmented_image = augmented_image.astype(np.float32)
            except:
                pass
            
            mask = mask.astype(np.float32)
            
            if len(mask.shape) == 2:
                mask = np.expand_dims(mask, axis=2)

            dec_brightness = 1
            if self.anomaly_type == [0]:
                patch_augmented = mask * augmented_image * dec_brightness
                patch_image = mask * image
                ssim_value = ssim(patch_augmented, patch_image, multichannel=True)
                if ssim_value > 0.85 and self.anomaly_type == [0, 1]:
                    perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                    augmented_image, perlin_threshold = per_anomaly(perlin_noise, anomaly_source_img, image)
                    augmented_image = augmented_image.astype(np.float32)
                    mask = perlin_threshold.astype(np.float32)

            has_anomaly = 1.0
            if np.sum(mask) == 0:
                has_anomaly = 0.0
            return augmented_image, mask, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))

        if len(anomaly_mask.shape) == 2:
            anomaly_mask = np.expand_dims(anomaly_mask, axis=2)

        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        anomaly_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        
        image, augmented_image, anomaly_mask, has_anomaly = \
            self.transform_image(self.image_paths[idx], self.anomaly_source_paths[anomaly_idx])

        return {
            'image': image, 
            'anomaly_mask': anomaly_mask,
            'augmented_image': augmented_image, 
            'has_anomaly': has_anomaly, 
            'idx': idx
        }