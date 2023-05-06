# -*- coding: utf-8 -*-
"""moco.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1kV32n5tWZ6a-SjCBBtjnXGQsdjuCzBHm

# Download
"""

import sys
import os
ROOT = "/vol/bitbucket/sx119/Contrastive-Medical-Image-Classification/"

"""# Import"""

import medmnist

import argparse
import math
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
import torch.utils.data
# import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models

import loader
import builder

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
SHUFFLED_SET = False
LEARNING_RATE = 0.05
END_LR = 0.05
LR_SCHEDULER = "linear"
# LR_SCHEDULER = "cos"
# LR_SCHEDULER = "multistep"
MULTI_STEPS = [12, 16] # used only in multi step decay
MOMENTUM = 0.9 # momentum of SGD
MOCO_MOMENTUM = 0.999 # momentum of moco
LOSS_TYPE = "self"
# LOSS_TYPE = "cate-ce"
# LOSS_TYPE = "binary-ce"
TRAIN_SET_RATIO = 0.9
MOCO_V2 = True
# ATTENTION_INFO = ("crop", 96, 96)
ATTENTION_INFO = ("mask", 8, 8)
COLOUR_AUG = True
# PRETRAIN_OPTIMISER = "SGD"
PRETRAIN_OPTIMISER = "Adam"
# PRETRAIN_OPTIMISER = "AdamW"
HEAD_LR = 0.01
# HEAD_OPTIMISER = "SGD"
HEAD_OPTIMISER = "Adam"
# HEAD_OPTIMISER = "AdamW"
PROJ_HEAD_EPOCH_NUM = 40
REMOVE_MLP = False

"""# Training helper functions"""

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, file, num_batches, meters, prefix=""):
        self.file = file
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.file.write('\t'.join(entries) + "\n")

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate based on schedule"""
    lr = args.lr
    if args.cos:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    else:  # stepwise lr schedule
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """
    NOT IN USE
    Computes the accuracy over the k top predictions for the specified values of k
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def update_accuracy_meters(losses, top1, top5, output, target, loss, step_size):
    """
    NOT IN USE
    Update loss, top1, top5 metrics for either train or validation

    Inputs:
      - step_size: parameter n for loss/top1/top5 meters
    """
    # acc1/acc5 are (K+1)-way contrast classifier accuracy
    # measure accuracy and record loss
    acc1, acc5 = accuracy(output, target, topk=(1, 5))
    losses.update(loss.item(), step_size)
    top1.update(acc1[0], step_size)
    top5.update(acc5[0], step_size)

def generate_loss_func(args):
  ce_loss_ = nn.CrossEntropyLoss(reduction="mean") # take softmax, sum CrossEntropy per sample, take mean
  binary_loss_ = nn.BCEWithLogitsLoss(reduction="mean") # take mean of per sample per pair binary Cross Entropy
  if args.gpu is not None:
    ce_loss_ = ce_loss_.cuda(args.gpu)
    binary_loss_ = binary_loss_.cuda(args.gpu)
  def multi_label_loss(prediction, target):
      """
      NOT IN USE
      Loss to handle multi-label classification when multiple positive image pairs exist.
      loss function used defined in args.loss_type
      
      Inputs: 
        - pretiction: shape: [bathc_size, 1 + k]
        - target: [batch_size] if self, otherwise [batch_size, 1 + K]
        
      Outputs:
        - scalar loss value for back propagate
      """
      
      if args.loss_type == "binary-ce":
        loss = binary_loss_(prediction, target).sum(dim=-1)
      elif args.loss_type in ["cate-ce", "self"]:
        loss = ce_loss_(prediction, target)
      
      return loss.mean()
  
  return multi_label_loss
  
def generate_datasets(args, distributed=True):
  """# Dataset"""

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
          transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize
      ]
  else:
      augmentation = [
          transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
          transforms.RandomApply([loader.GaussianBlur([.1, 2.])], p=0.5),
          transforms.RandomHorizontalFlip(),
          transforms.ToTensor(),
          normalize
      ]

  # # legacy split dataset operation, should load splitted data instead
  # train_dataset = medmnist.PathMNIST("train", download=False, root=args.data)

  # combined_set = torch.utils.data.ConcatDataset([train_dataset, val_dataset])
  # train_dataset, val_dataset = torch.utils.data.random_split(combined_set, [len(train_dataset), len(val_dataset)])

  # pretrain_len = int(len(train_dataset) * TRAIN_SET_RATIO)
  # pretrain_set, pretrain_val_set = torch.utils.data.random_split(train_dataset, [pretrain_len, len(train_dataset) - pretrain_len])
  # torch.save(pretrain_set, "pretrain_set.data")
  # torch.save(pretrain_val_set, "pretrain_val_set.data")

  # val_dataset = medmnist.PathMNIST("val", download=False, root=args.data)

  # dev_train_len = int(len(val_dataset) * TRAIN_SET_RATIO)
  # dev_train_set, dev_val_set = torch.utils.data.random_split(val_dataset, [dev_train_len, len(val_dataset) - dev_train_len])
  # torch.save(dev_train_set, "dev_train_set.data")
  # torch.save(dev_val_set, "dev_val_set.data")

  # # proper dataset loading, by loading pre-splitted data
  if SHUFFLED_SET:
      prefix = "shuffled_"
  else:
      prefix = ""
  pretrain_set = loader.MOCODataset(args.data + f"/{prefix}pretrain_set.data", augmentation)
  pretrain_val_set = loader.MOCODataset(args.data + f"/{prefix}pretrain_val_set.data", augmentation)

  dev_train_set = loader.MOCODataset(args.data + f"/{prefix}dev_train_set.data", augmentation)
  dev_val_set = loader.MOCODataset(args.data + f"/{prefix}dev_val_set.data", augmentation)
  test_dataset = medmnist.PathMNIST("test", download=False, root=ROOT + "/datasets/", 
                                    transform=transforms.Compose([
                                      transforms.Resize(224),
                                      transforms.ToTensor(),
                                      normalize
                                    ]))

  if distributed:
      pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_set)
      pretrain_val_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_val_set)
  else:
      pretrain_sampler = pretrain_val_sampler = None

  pretrain_loader = torch.utils.data.DataLoader(
      pretrain_set, batch_size=args.batch_size, shuffle=(pretrain_sampler is None), 
      pin_memory=True, drop_last=True, num_workers=4, sampler=pretrain_sampler)
  pretrain_val_loader = torch.utils.data.DataLoader(
      pretrain_val_set, batch_size=2 * args.batch_size, shuffle=False, 
      pin_memory=True, drop_last=True, num_workers=4, sampler=pretrain_val_sampler)

  dev_train_loader = torch.utils.data.DataLoader(
      dev_train_set, batch_size=args.batch_size, shuffle=True, 
      pin_memory=True, drop_last=True, num_workers=4)
  dev_val_loader = torch.utils.data.DataLoader(
      dev_val_set, batch_size=2 * args.batch_size, shuffle=False, 
      pin_memory=True, drop_last=True, num_workers=4)

  test_loader = torch.utils.data.DataLoader(
      test_dataset, batch_size=2 * args.batch_size, shuffle=False, 
      pin_memory=True, drop_last=True, num_workers=4)

  # print(f"pretrain size: {len(pretrain_set)}\npretrain validation size: {len(pretrain_val_set)}\ndev train size: {len(dev_train_set)}\ndev val size: {len(dev_val_set)}\ntest size: {len(test_dataset)}")

  return pretrain_loader, pretrain_val_loader, dev_train_loader, dev_val_loader, test_loader, pretrain_sampler

def generate_trial_name():
    return f"epochs{EPOCH_NUM}_shuffled{SHUFFLED_SET}_lr-pretrain{LEARNING_RATE}-{END_LR}-{LR_SCHEDULER}-decay-{'-'.join(map(str, MULTI_STEPS))}" \
    f"-head{HEAD_LR}_att-info{'-'.join([str(x) for x in ATTENTION_INFO])}_aug-colour{COLOUR_AUG}_optimizer-pretrain{PRETRAIN_OPTIMISER}-head{HEAD_OPTIMISER}_remove-mlp{REMOVE_MLP}"

def attention_locating(grad):
  ''' 
  get centered at pixel with max grad value
  
  Input: 
  - grad: shape (batch_size, 3, H, W)
  - h, w: int

  Output:
  - center_z, center_x, center_y: shape: (batch_size, )
  '''
  b, c, h, w = grad.shape
  index = grad.reshape((b, -1)).argmax(dim=-1)
  center_z = torch.div(index, (w * h), rounding_mode='trunc')
  center_h, center_w = torch.div((index - center_z * w * h), w, rounding_mode='trunc'), ((index - center_z * w * h)) % w
  return center_z, center_h, center_w

def attention_crop(img, attention, h, w):
  '''
  crop the region around the max grad, cropped region of size (h, w)

  Input:
  - img, attention: shape: (batch_size, 3, H, W)

  Output: 
  - shape: img with no grad, shape: (batch_size, 3, H, W)
  '''
  center_z, center_h, center_w = attention_locating(attention)
  h_axis = torch.from_numpy(np.linspace(center_h.cpu() - h / 2, center_h.cpu() + h / 2, img.shape[2])).cuda(args.gpu, non_blocking=True).T / img.shape[2]
  w_axis = torch.from_numpy(np.linspace(center_w.cpu() - w / 2, center_w.cpu() + w / 2, img.shape[3])).cuda(args.gpu, non_blocking=True).T / img.shape[3]
  h_axis = h_axis[:, :, None, None].repeat(1,1,img.shape[3],1)
  w_axis = w_axis[:, None, :, None].repeat(1,img.shape[2],1,1)
  grid = torch.cat([w_axis, h_axis], dim=-1)
  grid = grid * 2 - 1
  return nn.functional.grid_sample(img, grid)

def attention_mask(img, attention, h, w):
  '''
  mask the region around the max grad

  Input:
  - img, attention: shape: (batch_size, 3, H, W)
  - h, w: number of patches along each side of image, H / h and W / w must be integer

  Output: 
  - shape: img with no grad, shape: (batch_size, 3, H, W)
  '''

  b, c, H, W = img.shape
  patch_size = (int(H / h), int(W / w))

  patches = nn.functional.unfold(attention.sum(dim=1, keepdim=True), kernel_size=patch_size, stride=patch_size)

  mask = torch.zeros((b, patches.shape[-1])).cuda(args.gpu, non_blocking=True)
  mask = mask.scatter(1, patches.sum(dim=1).argmax(dim=-1)[:, None], 1)

  mask = mask[:, None, :].repeat_interleave(patches.shape[1], dim=1)
  mask = nn.functional.fold(mask, (H, W), kernel_size=patch_size, stride=patch_size)
  
  img = img.masked_fill(torch.gt(mask, 0), 0)
  
  return img

def main_worker(rank, world_size, args):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  # initialize the process group
  torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)
  torch.cuda.set_device(rank)

  model = builder.MoCo(
      models.__dict__[args.arch],
      args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
  # print(model)
  # model = torch.load("./71727_epochs15_batch128_lr0.03_momentum0.9_moco-momentum0.999_loss-typeself_V2True_att-infomask-8-8.pickle")
  
  pretrain_loader, pretrain_val_loader, dev_train_loader, dev_val_loader, test_loader, pretrain_sampler = generate_datasets(args, distributed=True)

  model = model.to(rank)
  model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

  criterion = generate_loss_func(args)

  if rank == 0:
      slurm_id = os.environ["SLURM_JOB_ID"]
      summary = open(f"{slurm_id}_{generate_trial_name()}.txt", "a")
  else:
      summary = None

  if PRETRAIN_OPTIMISER == "SGD":
      optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
  elif PRETRAIN_OPTIMISER == "Adam":
      optimizer = torch.optim.Adam(model.parameters(), args.lr)
  elif PRETRAIN_OPTIMISER == "AdamW":
      optimizer = torch.optim.AdamW(model.parameters(), args.lr)
  else:
    raise NotImplementedError("pretrain optimiser: " + PRETRAIN_OPTIMISER)

  if LR_SCHEDULER == "linear":
      lr_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=END_LR / args.lr, total_iters=EPOCH_NUM)
  elif LR_SCHEDULER == "cos":
      lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCH_NUM // 3, eta_min=END_LR)
  elif LR_SCHEDULER == "multistep":
      lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=MULTI_STEPS)
  else:
      raise NotImplementedError("lr scheduler" + LR_SCHEDULER)

  if args.gpu is not None:
    cudnn.benchmark = True

  for epoch in range(args.start_epoch, args.epochs):
      pretrain_sampler.set_epoch(epoch)
      adjust_learning_rate(optimizer, epoch, args)

      # train for one epoch
      batch_time = AverageMeter('Time', ':6.3f')
      data_time = AverageMeter('Data', ':6.3f')
      losses = AverageMeter('TrainLoss', ':.4e')
      # top1 = AverageMeter('Acc@1', ':6.2f')
      # top5 = AverageMeter('Acc@5', ':6.2f')
      val_losses = AverageMeter('ValLoss', ':.4e')
      # val_top1 = AverageMeter('ValAcc@1', ':6.2f')
      # val_top5 = AverageMeter('ValAcc@5', ':6.2f')
      progress = ProgressMeter(
          summary,
          len(pretrain_loader),
          # [batch_time, data_time, losses, top1, top5, val_losses, val_top1, val_top5],
          [batch_time, data_time, losses, val_losses],
          prefix="Epoch: [{}]".format(epoch))

      # switch to train mode
      model.train()

      end = time.time()
      with tqdm.tqdm(pretrain_loader, unit="batch") as tepoch: 
        tepoch = pretrain_loader
        for i, (images, labels) in enumerate(tepoch):
          # set label, if no label given, positive pair is image itself
          if args.loss_type == 'self':
            labels = None

          # measure data loading time
          data_time.update(time.time() - end)

          if args.gpu is not None:
              # images[0] = images[0].cuda(args.gpu, non_blocking=True)
              # images[1] = images[1].cuda(args.gpu, non_blocking=True)
              images[0] = images[0].to(rank)
              images[1] = images[1].to(rank)

          # compute output
          output, target = model(im_q=images[0], im_k=images[1], labels=labels)
          loss = criterion(output, target)

          # compute gradient and do SGD step
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()

          # update_accuracy_meters(losses, top1, top5, output, target, loss, images[0].size(0))
          losses.update(loss.item(), images[0].size(0))

          # measure elapsed time
          batch_time.update(time.time() - end)
          end = time.time()

          # log performance
          if i % args.print_freq == 0 and not i == 0:
            
            acc_val_loss = 0
            with torch.no_grad():
              model.eval()
              # evaluate on validation set
              for (images, labels) in pretrain_val_loader:
                if args.loss_type == 'self':
                  labels = None
                if args.gpu is not None:
                  images[0] = images[0].to(rank)
                  images[1] = images[1].to(rank)
                output, target = model(im_q=images[0], im_k=images[1], labels=labels, train=False)
                val_loss = criterion(output, target)
                acc_val_loss += val_loss.item()
              
              model.train()
            
            acc_val_loss /= len(pretrain_val_loader)
            _results = [None for _ in range(world_size)]
            torch.distributed.all_gather_object(_results, acc_val_loss)
            val_losses.update(sum(_results) / len(_results), 1) # only updated once

            if rank == 0:
              progress.display(i)

      lr_scheduler.step()

      # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
      #         and args.rank % ngpus_per_node == 0):
      #     save_checkpoint({
      #         'epoch': epoch + 1,
      #         'arch': args.arch,
      #         'state_dict': model.state_dict(),
      #         'optimizer' : optimizer.state_dict(),
      #     }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))

  if rank == 0:
    torch.save(model.state_dict(), f"{os.environ['SLURM_JOB_ID']}_{generate_trial_name()}.pickle")
    mlp_training(args, model, summary)

"""# Quantitative evaluation"""

def mlp_training(args, model, summary):
  # model.load_state_dict(torch.load("./73768_epochs1_shuffledFalse_lr-pretrain0.075-0.075-linear-decay-12-16-head0.01_aug-colourTrue_optimizer-pretrainAdam-headAdam_remove-mlpTrue.pickle"))

  EMB_SIZE = 128
  if REMOVE_MLP:
      model.module.encoder_q.fc = nn.Identity()
      EMB_SIZE = 2048

  def extract_data(loader):
    data_tensor = torch.zeros((0, EMB_SIZE)).cuda(args.gpu, non_blocking=True)
    label_tensor = torch.zeros((0, 1), dtype=torch.int32).cuda(args.gpu, non_blocking=True)
    with torch.no_grad():
      with tqdm.tqdm(loader, unit="batch") as tepoch: 
        tepoch = loader
        for images, labels in tepoch:
          images[0] = images[0].cuda(args.gpu, non_blocking=True)
          labels = labels.cuda(args.gpu, non_blocking=True)
          
          q = model.module.encoder_q(images[0])  # queries: NxC
          q = nn.functional.normalize(q, dim=1)

          data_tensor = torch.vstack([data_tensor, q])
          label_tensor = torch.vstack([label_tensor, labels])

    feature_dataset = torch.utils.data.TensorDataset(data_tensor, label_tensor)
    feature_loader = torch.utils.data.DataLoader(
      feature_dataset, batch_size=2 * args.batch_size, shuffle=True, drop_last=True)
    
    return feature_loader
  
  pretrain_loader, pretrain_val_loader, dev_train_loader, dev_val_loader, test_loader, pretrain_sampler = generate_datasets(args, distributed=False)


  model.eval()
  eval_set_info = [("dev_train_loader", dev_train_loader, dev_val_loader, PROJ_HEAD_EPOCH_NUM * 8), ("pretrain_loader", pretrain_loader, pretrain_val_loader, PROJ_HEAD_EPOCH_NUM)]
  # eval_set_info = [("dev_train_loader", dev_train_loader, dev_val_loader, PROJ_HEAD_EPOCH_NUM * 8)]
  for eval_loader_name, eval_loader, eval_val_loader, eval_epoch_num in eval_set_info:
      head_train_loader = extract_data(eval_loader)
      head_val_loader = extract_data(eval_val_loader)

      classification_head = nn.Linear(EMB_SIZE, 9).cuda(args.gpu)
      
      if HEAD_OPTIMISER == "SGD":
          head_optimizer = torch.optim.SGD(classification_head.parameters(), HEAD_LR, momentum=0.9, weight_decay=1e-4)
      elif HEAD_OPTIMISER == "Adam":
          head_optimizer = torch.optim.Adam(classification_head.parameters(), HEAD_LR)
      elif HEAD_OPTIMISER == "AdamW":
          head_optimizer = torch.optim.AdamW(classification_head.parameters(), HEAD_LR)
      else:
        raise NotImplementedError("projection head optimiser: " + HEAD_OPTIMISER)

      ce_loss = nn.CrossEntropyLoss(reduction="mean")

      classification_head.train()
      for epoch in range(eval_epoch_num):
        with tqdm.tqdm(head_train_loader, unit="batch") as tepoch: 
          tepoch = head_train_loader
          for i, (q, labels) in enumerate(tepoch):
            y_hat = classification_head(q)

            l = ce_loss(y_hat, labels.squeeze())

            head_optimizer.zero_grad()
            l.backward()
            head_optimizer.step()

            if i % 10 == 0 and not i == 0:
              classification_head.eval()

              acc_val_loss = 0
              with torch.no_grad():
                for (q, labels) in head_val_loader:                
                  y_hat = classification_head(q)
                  acc_val_loss += ce_loss(y_hat, labels.squeeze())
              
              classification_head.train()

              summary.write(f"classification loss: {eval_loader_name}: epoch {epoch}[{i}]{l.item()}, val avg loss: {acc_val_loss / len(eval_val_loader)}\n")

      classification_head.eval()
      acc_l = 0
      confusion_matrix = torch.zeros((9, 9))
      with torch.no_grad():
        for (img, label) in test_loader:
          img = img.cuda(args.gpu, non_blocking=True)
          label = label.cuda(args.gpu, non_blocking=True)

          q = model.module.encoder_q(img)
          q = nn.functional.normalize(q, dim=1)
          label_hat = classification_head(q)
          l = ce_loss(label_hat, label.squeeze())
                  
          acc_l += l

          for i in range(label.shape[0]):
            confusion_matrix[label[i].item(), label_hat[i].argmax().item()] += 1

      acc_f1 = 0
      for i in range(confusion_matrix.shape[0]):
        recall = confusion_matrix[i, i] / confusion_matrix[i].sum()
        precision = confusion_matrix[i, i] / confusion_matrix[:, i].sum()
        acc_f1 += 2 / (1 / precision + 1 / recall)

      summary.write(f"test set loss: {acc_l / len(test_loader)}, macro F1: {acc_f1 / confusion_matrix.shape[0]}\n")
      summary.write("\n")

      slurm_id = os.environ["SLURM_JOB_ID"]
      torch.save(confusion_matrix, f"{slurm_id}_{generate_trial_name()}_{eval_loader_name}_confusion_matrix.pickle")
      torch.save(classification_head, f"{slurm_id}_{generate_trial_name()}_head_{eval_loader_name}.pickle")

if __name__ == '__main__':
    arg_command = \
    f"--epochs_{EPOCH_NUM}_-b_{BATCH_SIZE}_--lr_{LEARNING_RATE}_--momentum_{MOMENTUM}_--moco-m_{MOCO_MOMENTUM}_--print-freq_50\
    _--loss-type_{LOSS_TYPE}_--gpu_0_{'--mlp_--aug-plus_--cos_' if MOCO_V2 else ''}{ROOT}./datasets".split("_")

    print(f"Running command {arg_command}")

    #!/usr/bin/env python
    # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet50)')
    # parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
    #                     help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    # parser.add_argument('--world-size', default=-1, type=int,
    #                     help='number of nodes for distributed training')
    # parser.add_argument('--rank', default=-1, type=int,
    #                     help='node rank for distributed training')
    # parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
    #                     help='url used to set up distributed training')
    # parser.add_argument('--dist-backend', default='nccl', type=str,
    #                     help='distributed backend')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    # parser.add_argument('--multiprocessing-distributed', action='store_true',
    #                     help='Use multi-processing distributed training to launch '
    #                          'N processes per node, which has N GPUs. This is the '
    #                          'fastest way to use PyTorch for either single node or '
    #                          'multi node data parallel training')
    # new argument proposed for medical image classification
    parser.add_argument('-lt', '--loss-type', default="binary-ce", type=str,
                        help='self if positive pairs are from the same image'
                        'cate-ce if categorical cross entropy loss is used for positive pairs from the same category'
                    'binary-ce if binary cross entropy loss is used for positive pairs from the same category')

    # moco specific configs:
    parser.add_argument('--moco-dim', default=128, type=int,
                        help='feature dimension (default: 128)')
    parser.add_argument('--moco-k', default=65536, type=int,
                        help='queue size; number of negative keys (default: 65536)')
    parser.add_argument('--moco-m', default=0.999, type=float,
                        help='moco momentum of updating key encoder (default: 0.999)')
    parser.add_argument('--moco-t', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')

    # options for moco v2
    parser.add_argument('--mlp', action='store_true',
                        help='use mlp head')
    parser.add_argument('--aug-plus', action='store_true',
                        help='use moco v2 data augmentation')
    parser.add_argument('--cos', action='store_true',
                        help='use cosine lr schedule')

    args = parser.parse_args(arg_command)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


    torch.multiprocessing.spawn(main_worker,
          args=(torch.cuda.device_count(),args),
          nprocs=torch.cuda.device_count(),
          join=True)