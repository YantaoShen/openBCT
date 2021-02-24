import argparse
import os
import random
import shutil
import time
import warnings
import pickle
from collections import OrderedDict
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import models as models
from utils.meters import AverageMeter
from evaluate import Evaluator, ClassifierGenerator
from utils.data.datasets import img_list_dataloader

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-f', '--use-feat', dest='use_feat', action='store_true',
                    help='evaluate model with feature')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='pre-trained model dir')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--old-fc', default=None, type=str, metavar='PATH',
                    help='old-classifier dir')
parser.add_argument('--n2o-map', default=None, type=str, metavar='PATH',
                    help='new to old label mapping dictionary dir')
parser.add_argument('--cross-eval', action='store_true',
                    help='conduct cross evaluation between diff models')
parser.add_argument('--old-arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                    ' | '.join(model_names) +
                    ' (default: resnet18)')
parser.add_argument('--old-checkpoint', default=None, type=str, metavar='PATH',
                    help='old backbone dir')
parser.add_argument('-g', '--generate-cls', action='store_true',
                    help='generate a pseudo classifier on current training set'
                         ' with a trained model')
parser.add_argument('--train-img-list', default=None, type=str, metavar='PATH',
                    help='train images txt')
parser.add_argument('--l2', action='store_true',
                    help='use l2 loss for compatible learning')
parser.add_argument('--lwf', action='store_true',
                    help='use l2 loss for compatible learning')
parser.add_argument('--val', action='store_true',
                    help='conduct validating when an epoch is finished')
parser.add_argument('--triplet', action='store_true',
                    help='use triplet loss for compatible learning')
parser.add_argument('--contra', action='store_true',
                    help='use contrastive loss for compatible learning')
parser.add_argument('--use-norm-sm', action='store_true',
                    help='use normed softmax for training')
parser.add_argument('--temp', default=0.05, type=float,
                    help='temperature for contrastive loss (default: 0.05)')
best_acc1 = 0.


def main():
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

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_trans = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(traindir, train_trans)

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    if args.train_img_list is None:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
            num_workers=args.workers, pin_memory=True, sampler=train_sampler)
        cls_num = len([d.name for d in os.scandir(traindir) if d.is_dir()])
    else:
        train_loader, cls_num = img_list_dataloader(traindir, args.train_img_list, train_trans,
                                                    args.distributed, batch_size=args.batch_size,
                                                    num_workers=args.workers)
        print('==> Using {} for loading data!'.format(args.train_img_list))
    if args.use_feat or args.cross_eval:
        cls_num = 0
        print('==> Using Feature distance, no classifier will be used!')
    else:
        print('==> Total {} classes!'.format(cls_num))

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        print("=> loading model from '{}'".format(args.checkpoint))
        model = models.__dict__[args.arch](old_fc=args.old_fc,
                                           use_feat=args.use_feat,
                                           num_classes=cls_num,
                                           norm_sm=args.use_norm_sm)
        checkpoint = torch.load(args.checkpoint)
        c_state_dict = OrderedDict()
        if 'state_dict' in checkpoint:
            checkpoint_dict = checkpoint['state_dict']
        else:
            checkpoint_dict = checkpoint
        for key, value in checkpoint_dict.items():
            if 'module.' in key:
                # remove 'module.' of data parallel
                name = key[7:]
                c_state_dict[name] = value
            else:
                c_state_dict[key] = value
        unfit_keys = model.load_state_dict(c_state_dict, strict=False)
        print('=> these keys in model are not in state dict: {}'.format(unfit_keys.missing_keys))
        print('=> these keys in state dict are not in model: {}'.format(unfit_keys.unexpected_keys))
        print("=> loading done!")
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](old_fc=args.old_fc,
                                           use_feat=args.use_feat or args.l2,
                                           num_classes=cls_num,
                                           norm_sm=args.use_norm_sm)
    if args.lwf:
        # According to Learning without Forgetting original paper (Li et.al. 2016),
        # the old classifier should be finetuned. However, it will not work for BCT.
        # So we freeze the old classifier.
        for para in model.old_fc.parameters():
            para.requires_grad = False

    if args.old_fc is not None:
        old_n = model.old_cls_num
    else:
        old_n = cls_num

    model = cudalize(model, ngpus_per_node, args)

    if args.old_checkpoint is not None:
        print("=> using old model '{}'".format(args.old_arch))
        print("=> loading old model from '{}'".format(args.old_checkpoint))
        old_model = models.__dict__[args.old_arch](use_feat=True,
                                                   num_classes=old_n)
        old_checkpoint = torch.load(args.old_checkpoint)

        oc_state_dict = OrderedDict()
        if 'state_dict' in old_checkpoint:
            old_checkpoint_dict = old_checkpoint['state_dict']
        else:
            old_checkpoint_dict = old_checkpoint
        for key, value in old_checkpoint_dict.items():
            if 'module.' in key:
                # remove 'module.' of data parallel
                name = key[7:]
                oc_state_dict[name] = value
            else:
                oc_state_dict[key] = value

        unfit_keys = old_model.load_state_dict(oc_state_dict, strict=False)
        print('=> these keys in model are not in state dict: {}'.format(unfit_keys.missing_keys))
        print('=> these keys in state dict are not in model: {}'.format(unfit_keys.unexpected_keys))
        print("=> loading done!")
        old_model = cudalize(old_model, ngpus_per_node, args)
    else:
        old_model = None

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        if args.cross_eval:
            print("==> cross test start...")
            validate(val_loader, model, criterion, args, old_model=old_model)
            return
        print("==> self test start...")
        validate(val_loader, model, criterion, args, cls_num=cls_num)
        return

    if args.generate_cls:
        print('==> generating the pseudo classifier on current training data')
        if args.train_img_list is not None:
            extract_loader, cls_num = img_list_dataloader(traindir, args.train_img_list,
                                                          transforms.Compose([transforms.Resize(256),
                                                                              transforms.CenterCrop(224),
                                                                              transforms.ToTensor(),
                                                                              normalize]),
                                                          distributed=args.distributed, batch_size=args.batch_size,
                                                          num_workers=args.workers, pin_memory=False,
                                                          )
        else:
            extract_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(traindir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=False)
        s_clsfier = generate_pseudo_classifier(extract_loader, old_model,
                                               cls_num=cls_num)
        if not os.path.isdir('./results/'):
            os.mkdir('./results/')
        with open(f'results/synth_clsfier.npy', 'wb') as f:
            np.save(f, s_clsfier)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args,
              old_model=old_model)

        if args.val:
            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, args, cls_num=cls_num)
        else:
            acc1 = 100.0  # always save the newest one
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            if not os.path.isdir('./results'):
                os.mkdir('./results')
            dirname = './results/' + '_'.join([str(args.arch),
                                               'dataset:' + str(args.train_img_list).split('/')[-1],
                                               'bct:' + str(args.old_fc).split('/')[-1],
                                               'lr:' + str(args.lr),
                                               'bs:' + str(args.batch_size),
                                               ])
            if not os.path.isdir(dirname):
                os.mkdir(dirname)
            print('==> Saving checkpoint to {}'.format(dirname))
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename=dirname + '/' + '_'.join(['epoch:' + str(epoch),
                                                           datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
                                                           'checkpoint.pth.tar'
                                                           ]))


def train(train_loader, model, criterion, optimizer, epoch, args, old_model=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    if args.old_fc is not None:
        n2o_map = np.load(args.n2o_map, allow_pickle=True).item() if args.n2o_map is not None else None
        old_losses = AverageMeter('Old Loss', ':.4e')
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, old_losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))
        if args.triplet:
            tri_losses = AverageMeter('Triplet Loss', ':.4e')
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, old_losses, tri_losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
        if args.contra:
            contra_losses = AverageMeter('Contrastive Loss', ':.4e')
            progress = ProgressMeter(
                len(train_loader),
                [batch_time, data_time, losses, old_losses, contra_losses, top1, top5],
                prefix="Epoch: [{}]".format(epoch))
    else:
        progress = ProgressMeter(
            len(train_loader),
            [batch_time, data_time, losses, top1, top5],
            prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    if old_model is not None:
        old_model.eval()  # fix old model

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        if args.l2:
            criterion = nn.MSELoss().cuda(args.gpu)
            output_feat = model(images)
            old_output_feat = old_model(images)
            loss = criterion(output_feat, old_output_feat)
            losses.update(loss.item(), images.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
            continue

        if args.old_fc is None:
            output = model(images)
            loss = criterion(output, target)
            old_loss = 0.
        else:
            output, old_output, output_feat = model(images)
            loss = criterion(output, target)
            valid_ind = []
            o_target = []
            if n2o_map is not None:
                for ind, t in enumerate(target):
                    if int(t) in n2o_map:
                        o_target.append(n2o_map[int(t)])
                        valid_ind.append(ind)
                if torch.cuda.is_available():
                    o_target = torch.LongTensor(o_target).cuda()
                else:
                    o_target = torch.LongTensor(o_target)
            else:
                # If there is no overlap, please use learning without forgetting,
                # or create pseudo old classifier with feature extraction.
                valid_ind = range(len(target))
                o_target = target
            if len(valid_ind) != 0:
                if args.lwf:
                    old_output_feat = old_model(images)
                    if torch.cuda.is_available():
                        pseudo_score = model.module.old_fc(old_output_feat)
                    else:
                        pseudo_score = model.old_fc(old_output_feat)
                    pseudo_label = F.softmax(pseudo_score, dim=1)
                    old_loss = -torch.sum(F.log_softmax(old_output[valid_ind]) * pseudo_label) / images.size(0)
                else:
                    old_loss = criterion(old_output[valid_ind], o_target)
            else:
                old_loss = 0.

            # if use triplet loss between new and old model
            if args.triplet:
                tri_criterion = nn.TripletMarginLoss().cuda(args.gpu)
                pos_old_output_feat = old_model(images)
                # find the hardest negative
                n = target.size(0)
                mask = target.expand(n, n).eq(target.expand(n, n).t())
                dist = torch.pow(output_feat, 2).sum(dim=1, keepdim=True).expand(n, n) + \
                       torch.pow(pos_old_output_feat, 2).sum(dim=1, keepdim=True).expand(n, n).t()
                dist = dist - 2 * torch.mm(output_feat, pos_old_output_feat.t())
                hardest_neg = []
                for index in range(n):
                    hardest_neg.append(pos_old_output_feat[dist[index][mask[index] == 0].argmin()])
                hardest_neg = torch.stack(hardest_neg)
                tri_loss = tri_criterion(output_feat, pos_old_output_feat, hardest_neg)

            # if use contrastive loss between old and new model
            if args.contra:
                old_output_feat = old_model(images)
                n = target.size(0)
                contra_loss = 0.
                old_output_feat = F.normalize(old_output_feat, dim=1)
                output_feat = F.normalize(output_feat, dim=1)
                for index in range(n):
                    pos_score = torch.mm(output_feat[index].unsqueeze(0), old_output_feat[index].unsqueeze(0).t())
                    neg_scores = torch.mm(output_feat[index].unsqueeze(0), old_output_feat[target[index] != target].t())
                    all_scores = torch.cat((pos_score, neg_scores), 1)
                    all_scores /= args.temp
                    # all positive samples are placed at 0-th position
                    p_label = torch.empty(1, dtype=torch.long).zero_().cuda()
                    contra_loss += criterion(all_scores, p_label)
                contra_loss /= n

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        if args.old_fc is not None and len(valid_ind) != 0:
            old_losses.update(old_loss.item(), len(valid_ind))
        if args.triplet:
            tri_losses.update(tri_loss.item(), images.size(0))
        if args.contra:
            contra_losses.update(contra_loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss = loss + old_loss
        if args.triplet:
            loss = tri_loss
        if args.contra:
            loss = contra_loss
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args, old_model=None, cls_num=1000):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    if args.use_feat:
        if args.cross_eval and old_model is not None:
            old_model.eval()
            evaluator = Evaluator(model, old_model)
        else:
            evaluator = Evaluator(model)
        top1, top5 = evaluator.evaluate(val_loader)
        print(' * Acc@1 {top1:.3f} Acc@5 {top5:.3f}'
              .format(top1=top1, top5=top5))
        return top1

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            if cls_num in target:
                print('Only have {} classes, test stop!'.format(cls_num))
                break

            # compute output
            if args.old_fc is None:
                output = model(images)
            else:
                output, _, _ = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '_'.join([filename.split('_epoch')[0], 'model_best.pth.tar']))


def cudalize(model, ngpus_per_node, args):
    """Select cuda or cpu mode on different machine"""
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    return model


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
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def generate_pseudo_classifier(train_loader, old_model, cls_num=1000):
    """Generate the pseudo classifier with new training data and old embedding model"""
    old_model.eval()
    cls_generator = ClassifierGenerator(old_model, cls_num)
    saved_classifier = cls_generator.generate_classifier(train_loader)
    return saved_classifier


if __name__ == '__main__':
    main()
