from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass, field


@dataclass
class StereoImage:
    name: str
    left_image: Image
    right_image: Image
    width: int = field(init=False)
    height: int = field(init=False)
    left_arr: np.ndarray = field(init=False, repr=False)
    right_arr: np.ndarray = field(init=False, repr=False)
    disparity: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.width, self.height = self.left_image.size
        self.left_arr = np.array(self.left_image, dtype=np.uint8)
        self.right_arr = np.array(self.right_image, dtype=np.uint8)
        self.disparity = np.zeros((self.width, self.height), np.uint8)

    def show(self):
        plt.figure(1, figsize=(15, 15))
        plt.subplot(121)
        plt.title(self.name + "_left")
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        plt.imshow(self.left_image, cmap='gray')

        plt.subplot(122)
        plt.title(self.name + "_right")
        plt.tick_params(axis='both', which='major', labelsize=10, labelbottom=False, bottom=False, top=False,
                        labeltop=True)
        plt.imshow(self.right_image, cmap='gray')

        plt.show()
