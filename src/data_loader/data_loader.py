import glob
from pathlib import Path
import cv2
from typing import List
import torch
from torchvision import transforms
import numpy as np

def list_folders(directory):
    return [f.name for f in Path(directory).iterdir() if f.is_dir()]

class MVTec:
    def __init__(
        self,
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
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.test = test
        self.image_transformations = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((resize_shape[0],resize_shape[1])),
            transforms.ToTensor(),
        ])
        
        # ====================
        # Read train images
        # ====================
        
        if test:
            self.paths = self.__get_images_paths(self.root_dir + "/test", file_extension)
        else:
            self.paths = self.__get_images_paths(self.root_dir + "/train", file_extension)

        # self.train_images = torch.stack([
        #     self.__transform_image(self.__load_image(path))
        #     for path in self.train_paths
        # ])
        
        # print(f"Found {len(self.train_images)} train normal images {self.train_images.size()}")
        
        # # ====================
        # # Read test normal images
        # # ====================

        # self.test_normal_paths = self.__get_images_paths(self.root_dir + "/test/good/", file_extension)
        
        # self.test_normal_images = torch.stack([
        #     self.__transform_image(self.__load_image(path))
        #     for path in self.test_normal_paths
        # ])
        
        # print(f"Found {len(self.test_normal_images)} test normal images {self.test_normal_images.size()}")

        # # ====================
        # # Read test anomaly images
        # # ====================
        
        # self.test_anomaly_paths = []
        
        # # Add only anomaly test images
        # for path in self.__get_images_paths(self.root_dir + "/test/", file_extension):
        #     if path.find("/good") == -1:
        #         self.test_anomaly_paths.append(path)
                
        # self.test_anomaly_paths = np.array(sorted(self.test_anomaly_paths))
        
        # self.test_anomaly_images = torch.stack([
        #     self.__transform_image(self.__load_image(path))
        #     for path in self.test_anomaly_paths
        # ])
        
        # print(f"Found {len(self.test_anomaly_images)} test anomaly images {self.test_anomaly_images.size()}")
        
        # # ====================
        # # Read test anomaly masks
        # # ====================
        
        # self.test_anomaly_masks_paths = self.__get_images_paths(root_dir + "/ground_truth", file_extension)
        
        # self.test_anomaly_masks = torch.stack([
        #     self.__transform_image(self.__load_image(path))
        #     for path in self.test_anomaly_masks_paths
        # ])
        
        # print(f"Found {len(self.test_anomaly_images)} test anomaly images masks {self.test_anomaly_images.size()}")
        
    def __get_images_paths(self, root_dir: str, file_extension: str) -> List[str]:
        paths = sorted(glob.glob(root_dir + f"*.{file_extension}"))

        if len(paths)==0: 
            paths = sorted(glob.glob(root_dir + f"/*/*.{file_extension}"))
        
        return paths
        
    def __transform_image(self, image):
        return self.image_transformations(image)
        
    def __load_image(self, path: str):
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image 
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        path = self.paths[idx]
        image = self.__load_image(path)

        image = self.__transform_image(image)
        
        mask = torch.zeros(3, image.size(1), image.size(2))
        
        if self.test and path.find("/good") == -1:
            mas_path = path.replace("/test/", "/ground_truth/")
            mas_path = mas_path.split(".")[0]+"_mask.png"
            
            mask = self.__load_image(mas_path)

            mask = self.__transform_image(mask)

        return image, mask