# -*- coding: utf-8 -*-
"""supervised.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y4-6uQf-Qt-fqiYjVa7UG0gbQWSS5Q1r

# Download
"""

WORKING_ENV = 'COLAB' # Can be LABS, COLAB or PAPERSPACE
assert WORKING_ENV in ['LABS', 'COLAB', 'LOCAL']

import sys
import os
if WORKING_ENV == 'COLAB':
  from google.colab import drive
  drive.mount('/content/drive/')
  !pip install medmnist
  !pip install torch
  !pip install gputil
  !pip install psutil
  !pip install humanize
  ROOT = "/content/drive/MyDrive/ColabNotebooks/med-contrastive-project/"
  sys.path.append(ROOT + "./supervised/")
  !nvidia-smi
  slurm_id = 0
elif WORKING_ENV == 'LABS':
  ROOT = "/vol/bitbucket/sx119/Contrastive-Medical-Image-Classification/"
  slurm_id = os.environ["SLURM_JOB_ID"]
else:
  ROOT = "/Users/xushitong/Contrastive-Medical-Image-Classification/"
  slurm_id = 0

"""# Import"""

import medmnist

import argparse
import builtins
import math
import os
import random
import shutil
import time
import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
# import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print("using device: ", dev)

# Import packages
import os,sys,humanize,psutil,GPUtil

# Define function
def mem_report():
  print("CPU RAM Free: " + humanize.naturalsize( psutil.virtual_memory().available ))
  
  GPUs = GPUtil.getGPUs()
  for i, gpu in enumerate(GPUs):
    print('GPU {:d} ... Mem Free: {:.0f}MB / {:.0f}MB | Utilization {:3.0f}%'.format(i, gpu.memoryFree, gpu.memoryTotal, gpu.memoryUtil*100))

mem_report()

"""# Hyperparameters"""

EPOCH_NUM = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.03
MOMENTUM = 0.9 # momentum of SGD
WEIGHT_DECAY = 1e-4 # weight decay for SGD
PRINT_FREQ = 10

trial_name = f"epoch{EPOCH_NUM}_batch{BATCH_SIZE}_lr{LEARNING_RATE}_momentum{MOMENTUM}_wd{WEIGHT_DECAY}"

print("trial name: " + trial_name)

"""# Dataset"""

class SupervisedDataset():
    """Dataset that applies given data augmentation to each image"""

    def __init__(self, path, augmentation):
        self.samples = torch.load(path)
        self.augmentation = transforms.Compose(augmentation)

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image, label = self.samples[index]
        return self.augmentation(image), label

# Data loading code
# traindir = os.path.join(args.data, 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
augmentation = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize
]

# # proper dataset loading, by loading pre-splitted data
pretrain_set = SupervisedDataset(ROOT + "/datasets/pretrain_set.data", augmentation=augmentation)
pretrain_val_set = SupervisedDataset(ROOT + "/datasets/pretrain_val_set.data", augmentation=augmentation)
val_dataset = medmnist.PathMNIST("val", download=False, root=ROOT + "/datasets/", 
                                   transform=transforms.Compose(augmentation))

pretrain_loader = torch.utils.data.DataLoader(
    pretrain_set, batch_size=BATCH_SIZE, shuffle=True, 
    pin_memory=True, drop_last=True)
pretrain_val_loader = torch.utils.data.DataLoader(
    pretrain_val_set, batch_size=2 * BATCH_SIZE, shuffle=False, 
    pin_memory=True, drop_last=True)

val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, 
    pin_memory=True, drop_last=True)

print(f"pretrain size: {len(pretrain_set)}\npretrain validation size: {len(pretrain_val_set)}\nvalidation size, {len(val_dataset)}")

"""# Train"""

if WORKING_ENV == 'LABS':
  summary = open(f"{slurm_id}_{trial_name}.txt", "a")
else:
  summary = sys.stdout


# create model
model = models.resnet50()
model.add_module("projection_head", nn.Linear(1000, 9))
model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction="mean")

optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=0.9, weight_decay=1e-4)

def train_func(img, label, train=True):
  # train/evaluate on given data for one mini batch 
  img = img.to(device)
  label = label.to(device)

  label_hat = model(img)
  l = criterion(label_hat, label.squeeze())

  if train:
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

  return l

def validate(loader):
  acc_l = 0
  with torch.no_grad():
    for (img, label) in loader:
      l = train_func(img, label, False)
      acc_l += l

  summary.write(f"validate loss: {acc_l / len(loader)}\n")

for epoch in range(EPOCH_NUM):
  acc_l = 0
  with tqdm.tqdm(pretrain_loader, unit="batch") as tepoch: 
    for i, (img, label) in enumerate(tepoch):
      l = train_func(img, label)
      acc_l += l.item()

      if i % PRINT_FREQ == 0 and i != 0:
        summary.write(f"Epoch {epoch}[{i}]: loss: {l.item()}[{acc_l / (i + 1)}]")
        validate(pretrain_val_loader)

        if not WORKING_ENV == 'LABS':
          tepoch.set_description(f"batch {i}")
          tepoch.set_postfix(loss=l.item())

torch.save(model, f"{slurm_id}_{trial_name}.pickle")
mem_report()