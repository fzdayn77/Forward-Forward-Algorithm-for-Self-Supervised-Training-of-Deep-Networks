#
# This source code is inspired from Pytorch Lightning SimCLR implementation :
#   https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html
#

import cv2
import numpy as np
from torchvision.transforms import transforms

class GaussianBlur(object):
    """
    Blurs the given image with separable convolution as described in the SimCLR paper
    (ArXiv, https://arxiv.org/abs/2002.05709).
    """

    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        # less than 50%
        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


class train_data_augmentation():
    """
    Implementation of the data augmentations on the trainig data
    as proposed in the SimCLR official paper
    """

    def __init__(
        self,
        size: int = 32,
        gaussian_blur: bool = False,
        jitter_strength: float = 1.,
        normalize = transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)) # Default Normalization
    ):
        self.jitter_strength = jitter_strength
        self.size = size
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            brightness=0.5 * self.jitter_strength,
            contrast=0.5 * self.jitter_strength,
            saturation=0.5 * self.jitter_strength,
            hue=0.1 * self.jitter_strength
        )

        data_transforms = [
            # Resizing images before any augmentation to a size of 32x32
            transforms.Resize(self.size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomResizedCrop(size=self.size),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2)
        ]

        # Adding Gaussian blur
        if self.gaussian_blur:
            data_transforms.append(GaussianBlur(kernel_size=int(0.1 * self.size), p=0.5))

        data_transforms.append(transforms.ToTensor())

        # Adding Normalization
        data_transforms.append(self.normalize)

        # Transformations on the training data
        self.train_transform = transforms.Compose(data_transforms)

    def __call__(self, x):
        x_i = self.train_transform(x)
        x_j = self.train_transform(x)
        return [x_i, x_j]


class test_data_augmentation():
    """
    Implementation of the data augmentations on the testing data
    as proposed in the SimCLR official paper
    """

    def __init__(
        self,
        size: int = 32,
        crop: bool = False,
        normalize = transforms.Normalize(mean=(0, 0, 0), std=(1, 1, 1)) # Default Normalization
    ):
        self.size = size
        self.crop = crop
        self.normalize = normalize

        data_transforms = [
            # Resizing images before any augmentation to a size of 32x32
            transforms.Resize(self.size)
        ]

        # Adding Crop
        if self.crop:
            data_transforms.append(transforms.RandomResizedCrop(size=self.size))

        data_transforms.append(transforms.ToTensor())

        # Adding Normalization
        data_transforms.append(self.normalize)

        # Transformations on the testing data
        self.test_transform = transforms.Compose(data_transforms)

    def __call__(self, x):
        x_i = self.test_transform(x)
        x_j = self.test_transform(x)
        return [x_i, x_j]