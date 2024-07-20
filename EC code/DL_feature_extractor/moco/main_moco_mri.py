#!/usr/bin/env python
import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import moco.loader
from moco.builder2 import MoCo
from monai.networks.nets.resnet import resnet50
from monai.transforms import (
    Compose,
    LoadImage,
    EnsureChannelFirst,
    Spacing,
    Lambda, RandSpatialCrop,
    RandGaussianSmooth,
    RandCropByPosNegLabel,
    CastToTyped,
    RandRotate90,
    RandAdjustContrast,
    RandAffine,
    RandGaussianNoise,
    ResizeWithPadOrCrop,
    RandFlip,
    Compose,
    adaptor,
    ToTensor,
    ResizeWithPadOrCrop
)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-data', default=r'E:\study\EC\subtype\Data_slice_3mri', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('-save', default=r'/.', help='path to save')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                             ' (default: resnet50)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N',
                        help='mini-batch size (default: 32)')
    parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--schedule', default=[300, 600], nargs='*', type=int,
                        help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 2)')
    parser.add_argument('--resume', default='E:\python\im4MEC-main\moco\checkpoint_0000.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')  # E:\python_DL\moco_v2_800ep_pretrain.pth.tar
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')

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
    parser.add_argument('--mlp', action='store_true', default=True,
                        help='use mlp head')
    parser.add_argument('--cos', action='store_true', default=True,
                        help='use cosine lr schedule')
    parser.add_argument('--warmup', default=10, type=int,
                        help='use a warmup with constant lr before cosine lr routine')
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = MoCo(
        models.__dict__[args.arch],
        args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)

    # model = MoCo(
    #     resnet50,
    #     args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
    print(model)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        model = model.cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            loc = 'cuda:{}'.format(args.gpu) if args.gpu is not None else None
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            state_dict = checkpoint['state_dict']

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # 删除 "module." 前缀
                else:
                    name = k
                new_state_dict[name] = v

            model.load_state_dict(new_state_dict)

            # model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'data')
    # augmentation = [
    #     transforms.ToTensor(),
    #     transforms.RandomResizedCrop(128, scale=(0.2, 1.)),
    #     transforms.RandomRotation(90),
    #     transforms.RandomVerticalFlip(),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomApply([
    #         transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
    #     transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])], p=0.5),
    # ]
    aug_ratio: float = 0.4
    augmentation = Compose([
        # Spacing(pixdim=(1, 1, 1)),
        # RandSpatialCrop(roi_size=(128, 128)),
        RandRotate90(
            prob=aug_ratio,
        ),
        RandFlip(
            prob=aug_ratio,
            spatial_axis=(0, 1),
        ),
        RandAdjustContrast(
            prob=aug_ratio,
            gamma=(0.8, 1.2)
        ),
        RandAffine(
            rotate_range=[math.pi / 30, math.pi / 30],
            shear_range=[0.1, 0.1],
            translate_range=[5, 5],
            scale_range=[0.1, 0.1],
        ),
        RandGaussianNoise(
            prob=aug_ratio / 2,
            mean=0.0,
            std=0.05),
        RandGaussianSmooth()
    ])

    train_dataset = datasets.DatasetFolder(
        traindir, nifty_loader, extensions=('.nii.gz'), transform=moco.loader.TwoCropsTransform(augmentation))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True)

    writer = SummaryWriter(args.save)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        losses, top1, top5 = train(train_loader, model, criterion, optimizer, epoch, args)

        writer.add_scalar('loss', losses, global_step=epoch)
        writer.add_scalar('top1', top1, global_step=epoch)
        writer.add_scalar('top5', top5, global_step=epoch)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename='checkpoint_{:04d}.pth.tar'.format(epoch))
    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()
    for i, (images, temp) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        output, target = model(im_q=images[0], im_k=images[1])
        loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images[0].size(0))
        top1.update(acc1[0], images[0].size(0))
        top5.update(acc5[0], images[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, top1.avg, top5.avg


def nifty_loader(path: str):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
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
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    if args.cos:
        lr = args.lr
        if epoch >= args.warmup:
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch - args.warmup) / (args.epochs - args.warmup)))
    else:
        lr = args.lr
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
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

if __name__ == '__main__':
    main()