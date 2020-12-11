import torch
from torchvision import datasets
from PIL import Image
import cv2
import numpy as np

class MNISTDataset(datasets.MNIST):
    def __init__(self, root, train=True, download=True, transform=None):
        super().__init__(root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            try: # albumentations
                image = np.expand_dims(image, -1).astype(np.uint8)
                imageA = self.transform(image=image)
                image = imageA["image"]
            except: # torchvision
                image = self.transform(image)
                
        return image, label