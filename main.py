from __future__ import print_function

import os
import time
import torch
import torchvision
import random
import argparse

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#data transforms
from torch.autograd import Variable
from torchvision import datasets, transforms

#data aug
from augmentation.autoaugment import CIFAR10Policy 
from augmentation.cutout import Cutout 
from augmentation.AugMix.AugMix import AugMixDataset 
from augmentation.RandAugment import RandAugment

#optim and activation
from adamod           import AdaMod
from optim.radam      import RAdam
from optim.lookahead  import Lookahead 
from optim.deepmemory import DeepMemory

from models.efficientnet_pytorch import EfficientNet
from metrics                     import AverageMeter, accuracy
from loss_func.cross_entropy     import CrossEntropyLoss

parser = argparse.ArgumentParser(description='Data Augmentation Techniques on CIFAR10 with PyTorch.')
# Data Augmentation Techniques
parser.add_argument('--cutout',  action='store_true', default=False, help='Using CutOut data augmentation technique.')
parser.add_argument('--autoaug',  action='store_true', default=False, help='Using AutoAugment data augmentation technique.')
parser.add_argument('--randaug', action='store_true', default=False, help='Using RandAugment data augmentation technique.')
parser.add_argument('--augmix',  action='store_true', default=False, help='Using AugMixt data augmentation technique.')
# Optimizers
parser.add_argument('--adamod', action='store_true', default=False, help='Use AdaMod optimizer')      # make default optimizer
parser.add_argument('--adalook', action='store_true', default=False, help='Use AdaMod+LookAhead optimizer')
parser.add_argument('--deepmemory', action='store_true', default=False, help='Use DeepMemory optimizer')
parser.add_argument('--ranger', action='store_true', default=False, help='Use RAdam+LookAhead optimizer')
# Others
parser.add_argument('--resume', '-r', action='store_true', default=False, help='resume training from checkpoint.')
parser.add_argument('--path', default='', type=str, help='path to checkpoint. pass augmentation name')
parser.add_argument('--epochs', '-e', default=50, type=int, help='Number of training epochs.')
parser.add_argument('--num_workers', default=4, type=int, help='Number of CPUs.')
parser.add_argument('--batch_size', '-bs', default=4, type=int, help='input batch size for training.')
parser.add_argument('--learning_rate', '-lr', default=1e-3, type=int, help='learning rate.')
parser.add_argument('--weight_decay', '-wd', default=0, type=float, help='weight decay.')
parser.add_argument('--print_freq', '-pf', default=100, type=int, help='Number of iterations to print out results')
parser.add_argument('--seed', default=65, type=int, help='random seed')
args = parser.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED']=str(seed)
    
set_seed(args.seed)

# Transform and Load Data
preprocess = transforms.Compose([
     transforms.ToTensor(),
     transforms.Normalize((0.4914, 0.4822, 0.4465), (.2023, .1994, .2010)),
])

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4, fill=128),
    preprocess
])

test_transform = preprocess

# Augmentation techniques
if args.cutout:
    print("\n==> Training model with cutout data augmentation technique...\n")
    preprocess.transforms.append(Cutout(n_holes=1, length=16))             # CutOut

if args.autoaug:
    print("\n==> Training model with automatic data augmentation technique...\n")
    preprocess.transforms.insert(0, CIFAR10Policy())                                 # AutoAugment

if args.randaug:
    print("\n==> Training model with random data augmentation technique...\n")
    train_transform.transforms.insert(0, RandAugment(1, 5))                    #RandAugment

train_data = datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
test_data= datasets.CIFAR10(root="./data", train=False, download=True, transform=test_transform)

if args.augmix:
    print("\n==> Training model with augmentation mix data augmentation technique...\n")
    train_transform.transforms.pop()
    train_data = AugMixDataset(train_data, preprocess, no_jsd=True)     # Augmix

train_loader = torch.utils.data.DataLoader(
                      train_data,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=True,
                      pin_memory=True
                      )
test_loader = torch.utils.data.DataLoader(test_data,
                      batch_size=args.batch_size,
                      num_workers=args.num_workers,
                      shuffle=False,
                      pin_memory=True
                      )

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=10)
model = model.to(device)
model = torch.nn.DataParallel(model)

# train from start
best_top1 = 0
start_epoch = 0

# criterion and optimizer
params = [p for p in model.parameters()]
criterion = CrossEntropyLoss(smooth_eps=0.1).to(device)

if args.adamod:
    print("\n Using AdaMod optimizer")
    optimizer = AdaMod(params, lr=args.learning_rate, weight_decay=args.weight_decay)

if args.deepmemory:
    print("\n Using DeepMemory optimizer")
    optimizer = Lookahead(DeepMemory(params, lr=args.learning_rate, weight_decay=args.weight_decay))
        
if args.adalook:
    print("\n Using AdaMod+LookAhead optimizer")
    optimizer = Lookahead(AdaMod(params, lr=args.learning_rate, weight_decay=args.weight_decay))

if args.ranger:
    print("\n Using AdaMod+LookAhead optimizer")
    optimizer = Lookahead(RAdam(params, lr=args.learning_rate, weight_decay=args.weight_decay))
    
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader))

# resume training from checkpoint
if args.resume:
    checkpoint = torch.load('./checkpoint/Baseline_'+args.path+'_ckpt.pth')
    model.module.load_state_dict(checkpoint['model_state_dict'], strict=False)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_loss = checkpoint['loss']
    best_top1 = checkpoint['top1']
    best_top5 = checkpoint['top5']

    print(f'Resume training with \n {best_top1}% Top-1 Accuracy, {best_top5}% Top-5 Accuracy, after training for {start_epoch-1} epochs.')

# Train Model
def train(train_loader, model, criterion, optimizer, epoch):
    print('\n Training model...\n')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.to(device)
        input_var = Variable(input)
        target_var = Variable(target)

        optimizer.zero_grad()

        # compute output
        output = model(input_var)

        def closure():
            output = model(input_var)
            loss = criterion(output, target_var)

            return loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(closure().item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        closure().backward()
        optimizer.step()

        lr_scheduler.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))

    print(' * Acc@1 {top1.avg:.3f} Acc@1 Error {top1_err:.3f}\n'
              ' * Acc@5 {top5.avg:.3f} Acc@5 Error {top5_err:.3f}'
              .format(top1=top1, top1_err=100-top1.avg, top5=top5, top5_err=100-top5.avg))

# Test model on test data
def test(test_loader, model, criterion, epoch):
    print('\n Running inference on test data...\n')
    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            target = target.to(device)
            input = input.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 5))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq//4 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(test_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))



        print(' * Acc@1 {top1.avg:.3f} Acc@1 Error {top1_err:.3f}\n'
              ' * Acc@5 {top5.avg:.3f} Acc@5 Error {top5_err:.3f}'
              .format(top1=top1, top1_err=100-top1.avg, top5=top5, top5_err=100-top5.avg))

        return top1, top5, losses

for epoch in range(start_epoch, args.epochs):
    train(train_loader, model, criterion, optimizer, epoch)
    top1, top5, losses = test(test_loader, model, criterion, epoch)

    if top1.avg > best_top1:
        print(f'\n *** Test accuracy improved from {best_top1}% to {top1.avg}%.\t Saving checkpoint\n')
        state = {
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'epoch': epoch,
            'loss': losses.avg,
            'top1': top1.avg,
            'top5': top5.avg}
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/Baseline_'+args.path+'_ckpt.pth')

        best_top1 = top1.avg
        
    else:
        print(f'\n *** Test accuracy did not improve from {best_top1}%')
