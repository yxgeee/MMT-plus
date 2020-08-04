from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
from apex import amp

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from visda import datasets
from visda import models
from visda.trainers import PreTrainer
from visda.evaluators import Evaluator
from visda.utils.data import transforms as T
from visda.utils.data.sampler import RandomMultipleGallerySampler
from visda.utils.data import IterLoader
from visda.utils.data.preprocessor import Preprocessor
from visda.utils.logging import Logger
from visda.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from visda.utils.lr_scheduler import WarmupMultiStepLR

start_epoch = best_mAP = 0

def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset

def get_train_loader(args, dataset, height, width, batch_size, workers,
                    num_instances, iters, epochs, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.ImageNetPolicy(iters*epochs),
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
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

def get_test_loader(args, dataset, height, width, batch_size, workers, testset=None):
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
    print("==> Load source-domain trainset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    source_classes = dataset_source.num_train_pids
    print("==> Load target-domain trainset")
    dataset_target = get_data('target_train', args.data_dir)
    print("==> Load target-domain valset")
    dataset_target_val = get_data('target_val', args.data_dir)

    test_loader_target = get_test_loader(args, dataset_target_val, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters, args.epochs)
    train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                        args.batch_size, args.workers, 0, iters, args.epochs)

    # Create model
    model = models.create(args.arch, num_features=args.features, norm=False,
                            dropout=args.dropout, num_classes=source_classes)
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

    # Load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        copy_state_dict(checkpoint['state_dict'], model)
        start_epoch = checkpoint['epoch']
        best_mAP = checkpoint['best_mAP']
        print("=> Start epoch {}  best mAP {:.1%}"
              .format(start_epoch, best_mAP))

    # Evaluator
    evaluator = Evaluator(model)
    if args.evaluate:
        print("Test on target domain:")
        evaluator.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery, rerank=args.rerank)
        return

    lr_scheduler = WarmupMultiStepLR(optimizer, args.milestones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

    # Trainer
    trainer = PreTrainer(model, source_classes, margin=args.margin, fp16=args.fp16)

    # Start training
    for epoch in range(start_epoch, args.epochs):
        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_target, optimizer,
                    train_iters=len(train_loader_source), print_freq=args.print_freq)

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            _, mAP = evaluator.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)

            is_best = mAP > best_mAP
            best_mAP = max(mAP, best_mAP)
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

            print('\n * Finished epoch {:3d} mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        lr_scheduler.step()

    print("Test on target domain:")
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery, rerank=args.rerank)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-training")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='personx',
                        choices=datasets.names())
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
