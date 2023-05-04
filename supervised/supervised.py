# -*- coding: utf-8 -*-
"""supervised.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Y4-6uQf-Qt-fqiYjVa7UG0gbQWSS5Q1r

# Download
"""

WORKING_ENV = 'LABS' # Can be LABS, COLAB or PAPERSPACE
assert WORKING_ENV in ['LABS', 'COLAB', 'LOCAL']

import sys
import os
if WORKING_ENV == 'COLAB':
  from google.colab import drive
  drive.mount('/content/drive/')
  # !pip install medmnist
  # !pip install torch
  # !pip install gputil
  # !pip install psutil
  # !pip install humanize
  # ROOT = "/content/drive/MyDrive/ColabNotebooks/med-contrastive-project/"
  # sys.path.append(ROOT + "./supervised/")
  # !nvidia-smi
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
from PIL import ImageFilter

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

EPOCH_NUM = 15
BATCH_SIZE = 128
SHUFFLED_SET = False
# OPTIMISER = "SGD"
OPTIMISER = "Adam"
# OPTIMISER = "AdamW"
LEARNING_RATE = 0.03
END_LR = 0.03
LR_SCHEDULER = "linear"
# LR_SCHEDULER = "cos"
MOMENTUM = 0.9 # momentum of SGD
WEIGHT_DECAY = 1e-4 # weight decay for SGD
ON_PRETRAINED = False # if trained on pre-training set or on validation set
if ON_PRETRAINED:
  PRINT_FREQ = 200
else:
  PRINT_FREQ = 20
COLOUR_AUG = True
NAIVE_RESNET = False # if mlp layer is on 1000 class from default resnet or on 2048 resnet cnn output

trial_name = f"epoch{EPOCH_NUM}_shuffled{SHUFFLED_SET}_lr{LEARNING_RATE}-{END_LR}_{LR_SCHEDULER}_on-pretrain{ON_PRETRAINED}_aug-colour{COLOUR_AUG}_optimizer{OPTIMISER}_naive-resnet{NAIVE_RESNET}"

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

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

# Data loading code
# traindir = os.path.join(args.data, 'train')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
if COLOUR_AUG:
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
else:
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
        transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]

# # proper dataset loading, by loading pre-splitted data
if SHUFFLED_SET:
    prefix = "shuffled_"
else:
    prefix = ""
pretrain_set = SupervisedDataset(ROOT + f"./datasets/{prefix}pretrain_set.data", augmentation)
pretrain_val_set = SupervisedDataset(ROOT + f"./datasets/{prefix}pretrain_val_set.data", augmentation)

dev_train_set = SupervisedDataset(ROOT + f"./datasets/{prefix}dev_train_set.data", augmentation)
dev_val_set = SupervisedDataset(ROOT + f"./datasets/{prefix}dev_val_set.data", augmentation)

test_dataset = medmnist.PathMNIST("test", download=False, root=ROOT + "/datasets/", 
                                  transform=transforms.Compose([
                                    transforms.Resize(224),
                                    transforms.ToTensor(),
                                    normalize
                                  ]))

pretrain_loader = torch.utils.data.DataLoader(
    pretrain_set, batch_size=BATCH_SIZE, shuffle=True, 
    pin_memory=True, drop_last=True)
pretrain_val_loader = torch.utils.data.DataLoader(
    pretrain_val_set, batch_size=2 * BATCH_SIZE, shuffle=False, 
    pin_memory=True, drop_last=True)

dev_train_loader = torch.utils.data.DataLoader(
    dev_train_set, batch_size=BATCH_SIZE, shuffle=True, 
    pin_memory=True, drop_last=True)
dev_val_loader = torch.utils.data.DataLoader(
    dev_val_set, batch_size=2 * BATCH_SIZE, shuffle=False, 
    pin_memory=True, drop_last=True)

test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, 
    pin_memory=True, drop_last=True)

print(f"pretrain size: {len(pretrain_set)}\npretrain validation size: {len(pretrain_val_set)}\ndev train size: {len(dev_train_set)}\ndev val size: {len(dev_val_set)}\ntest size: {len(test_dataset)}")

"""# Train"""

if WORKING_ENV == 'LABS':
  summary = open(f"{slurm_id}_{trial_name}.txt", "a")
else:
  summary = sys.stdout


# create model
model = models.resnet50()
if NAIVE_RESNET:
    model.add_module("projection_head", nn.Linear(1000, 9))
else:
    model.fc = nn.Identity()
    model.add_module("projection_head", nn.Linear(2048, 9))
model = model.to(device)

criterion = nn.CrossEntropyLoss(reduction="mean")

if OPTIMISER == "SGD":
    optimizer = torch.optim.SGD(model.parameters(), LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
elif OPTIMISER == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
elif OPTIMISER == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), LEARNING_RATE)
else:
    raise NotImplementedError("optimiser: " + OPTIMISER)

if LR_SCHEDULER == "linear":
    lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=END_LR / LEARNING_RATE, total_iters=EPOCH_NUM)
elif LR_SCHEDULER == "cos":
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_NUM // 3, eta_min=END_LR)
else:
    raise NotImplementedError("lr scheduler" + LR_SCHEDULER)

def train_func(img, label, train=True, return_y=False):
  # train/evaluate on given data for one mini batch 
  img = img.to(device)
  label = label.to(device)

  if train:
    model.train()
  else:
    model.eval()

  label_hat = model(img)
  l = criterion(label_hat, label.squeeze())

  if train:
    optimizer.zero_grad()
    l.backward()
    optimizer.step()

  if return_y:
    return l, label_hat
  return l

for epoch in range(EPOCH_NUM):
  acc_l = 0
  if ON_PRETRAINED:
    loader = pretrain_loader
    val_loader = pretrain_val_loader
  else:
    loader = dev_train_loader
    val_loader = dev_val_loader
  with tqdm.tqdm(loader, unit="batch") as tepoch: 
    if WORKING_ENV == "LABS":
      tepoch = loader
    for i, (img, label) in enumerate(tepoch):
      l = train_func(img, label)
      acc_l += l.item()

      if i % PRINT_FREQ == 0 and i != 0:
        if WORKING_ENV == 'LABS':
          print(f"batch {i} loss: {l.item()}")
        else:
          tepoch.set_description(f"batch {i}")
          tepoch.set_postfix(loss=l.item())

        acc_val_l = 0
        with torch.no_grad():
          for img, label in val_loader:
            l = train_func(img, label, False)
            acc_val_l += l.item()

        summary.write(f"Epoch {epoch}[{i}]: loss: {l.item()}({acc_l / (i + 1)}), val loss: {acc_val_l / len(val_loader)} lr {lr_scheduler.get_last_lr()}\n")
    lr_scheduler.step()

torch.save(model, f"{slurm_id}_{trial_name}.pickle")
mem_report()

"""# Quantitative Evaluation"""

acc_l = 0
confusion_matrix = torch.zeros((9, 9))
with torch.no_grad():
  for (img, label) in test_loader:
    l, label_hat = train_func(img, label, train=False, return_y=True)
    acc_l += l

    for i in range(label.shape[0]):
      confusion_matrix[label[i].item(), label_hat[i].argmax().item()] += 1

acc_f1 = 0
for i in range(confusion_matrix.shape[0]):
  recall = confusion_matrix[i, i] / confusion_matrix[i].sum()
  precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
  acc_f1 += 2 / (1 / precision + 1 / recall)

torch.save(confusion_matrix, f"{slurm_id}_{trial_name}_confusion_matrix.pickle")
summary.write(f"test set loss: {acc_l / len(test_loader)}, macro F1: {acc_f1 / confusion_matrix.shape[0]}\n")

