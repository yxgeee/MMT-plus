from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time

from apex import amp

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from visda import datasets
from visda import models
from visda.trainers import CameraTrainer
from visda.evaluators import Evaluator
from visda.utils.data import transforms as T
from visda.utils.data.sampler import RandomMultipleGallerySampler
from visda.utils.data import IterLoader
from visda.utils.data.preprocessor import Preprocessor
from visda.utils.logging import Logger
from visda.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from visda.utils.lr_scheduler import WarmupMultiStepLR
from visda.utils.meters import AverageMeter
from visda.evaluation_metrics import accuracy

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, epochs, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
	         T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                            batch_size=batch_size, num_workers=workers, sampler=sampler,
                            shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader

def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader

def validate(model, data_loader, print_freq=50):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    precisions = AverageMeter()

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, _, camids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            imgs = imgs.cuda()
            camids = camids.cuda()

            outputs = model(imgs)
            probs = model.module.classifier(outputs)
            prec, = accuracy(probs.data, camids.data)
            precisions.update(prec[0])

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              precisions.val, precisions.avg))

    return precisions.avg

def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)


def main_worker(args):
    global start_epoch, best_mAP
    cudnn.benchmark = True

    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    else:
        log_dir = osp.dirname(args.resume)
        sys.stdout = Logger(osp.join(log_dir, 'log_test.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    print("==> Load target-domain trainset")
    dataset_target = get_data('target_train', args.data_dir)
    print("==> Load target-domain valset")
    dataset_target_val = get_data('target_val', args.data_dir)

    test_loader_target = get_test_loader(dataset_target_val, args.height, args.width, args.batch_size, args.workers)
    train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                        args.batch_size, args.workers, 0, iters, args.epochs)

    # Create model
    model_kwargs = {'num_features':args.features, 'norm':False,
                    'dropout':args.dropout, 'num_classes':dataset_target.num_train_cams,}
                    # 'metric':args.metric, 's':args.metric_s, 'm':args.metric_m}
    model = models.create(args.arch, **model_kwargs)
    model.cuda()

    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    model = nn.DataParallel(model)

    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    trainer = CameraTrainer(model, dataset_target.num_train_cams, margin=args.margin, fp16=args.fp16)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_target, optimizer,
                    train_iters=len(train_loader_target), print_freq=args.print_freq)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            mAP = validate(model, test_loader_target)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d}  accuracy: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Camera classification training")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet50',
                        choices=models.names())
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('-m', '--metric', type=str, default='arc')
    parser.add_argument('-ms', '--metric-s', type=float, default=64)
    parser.add_argument('-mm', '--metric-m', type=float, default=0.35)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[40, 70], help='milestones for the learning rate decay')
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--eval-step', type=int, default=40)
    parser.add_argument('--rerank', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    parser.add_argument('--fp16', action='store_true', help="training only")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
