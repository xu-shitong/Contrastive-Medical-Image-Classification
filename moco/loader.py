# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import torch
import torchvision.transforms as transforms

class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class MOCODataset():
    """Dataset that applies given data augmentation to each image"""

    def __init__(self, path, augmentation):
        self.samples = torch.load(path)
        self.augmentation = TwoCropsTransform(transforms.Compose(augmentation))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image, label = self.samples[index]
        return self.augmentation(image), label