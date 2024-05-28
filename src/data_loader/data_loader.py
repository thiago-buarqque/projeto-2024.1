import glob
from pathlib import Path
import cv2
from typing import List
import torch
from torchvision import transforms
import numpy as np
import torch.nn.functional as F
import imgaug.augmenters as iaa

def list_folders(directory):
    return [f.name for f in Path(directory).iterdir() if f.is_dir()]

class MVTec:
    def __init__(
        self,
        class_name: str,
        root_dir: str,
        file_extension = "png",
        resize_shape: List[int] = [512, 512],
        test: bool = False):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.class_name = class_name
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.test = test
        self.image_transformations = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.Resize((resize_shape[0],resize_shape[1])),
            transforms.ToTensor(),
        ])
        
        self.augmenter = iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
            iaa.pillike.EnhanceSharpness(),
        ])
        
        if test:
            self.paths = self.__get_images_paths(self.root_dir + "/test", file_extension)
        else:
            self.paths = self.__get_images_paths(self.root_dir + "/train", file_extension)
        
    def __get_images_paths(self, root_dir: str, file_extension: str) -> List[str]:
        paths = sorted(glob.glob(root_dir + f"*.{file_extension}"))

        if len(paths)==0: 
            paths = sorted(glob.glob(root_dir + f"/*/*.{file_extension}"))
        
        return paths
        
    def __transform_image(self, image):
        return self.image_transformations(self.augmenter(image=image))
        
    def __load_image(self, path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        
        return image 
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.__load_image(path)

        image = self.__transform_image(image)
        
        mask = torch.zeros(3, image.size(1), image.size(2))
        
        has_anomaly = False
        if self.test and path.find("/good") == -1:
            mas_path = path.replace("/test/", "/ground_truth/")
            mas_path = mas_path.split(".")[0]+"_mask.png"
            
            mask = self.__load_image(mas_path)

            mask = self.__transform_image(mask)
            
            has_anomaly = True

        return image, mask, has_anomaly