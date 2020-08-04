from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import shutil
import collections

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from visda import datasets
from visda import models
from visda.evaluators import Evaluator, extract_features
from visda.utils.data import transforms as T
from visda.utils.data import IterLoader
from visda.utils.data.sampler import RandomMultipleGallerySampler, ShuffleBatchSampler
from visda.utils.data.preprocessor import Preprocessor
from visda.utils.logging import Logger
from visda.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from visda.utils.osutils import mkdir_if_missing

from visda.sda.options.train_options import TrainOptions
from visda.sda.models.sda_model import SDAModel
from visda.sda.util.visualizer import Visualizer
from visda.sda.models import networks


start_epoch = best_mAP = 0

def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset

def get_train_loader(dataset, height, width, batch_size, workers,
                    num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    transformers = [T.Resize((height, width), interpolation=3),
                     T.RandomHorizontalFlip(p=0.5),
                     T.Pad(10),
                     T.RandomCrop((height, width)),
                     T.ToTensor(),
                     normalizer]
    train_transformer = T.Compose(transformers)

    train_set = dataset.train if trainset is None else trainset
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
        train_loader = IterLoader(
                    DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                            transform=train_transformer),
                                num_workers=workers, pin_memory=True,
                                batch_sampler=ShuffleBatchSampler(sampler, batch_size, True)), length=iters)
    else:
        train_loader = IterLoader(
                    DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                            transform=train_transformer),
                                batch_size=batch_size, num_workers=workers, sampler=None,
                                shuffle=True, pin_memory=True, drop_last=True), length=iters)

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


def main():
    args = TrainOptions().parse()   # get training argsions
    args.checkpoints_dir = args.logs_dir

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        cudnn.deterministic = True

    mkdir_if_missing(args.logs_dir)

    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP

    args.gpu = None
    args.rank = 0

    visualizer = Visualizer(args)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create data loaders
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain trainset")
    dataset_source = get_data('personx', args.data_dir)
    print("==> Load target-domain trainset")
    dataset_target = get_data('target_train', args.data_dir)
    print("==> Load target-domain valset")
    dataset_target_val = get_data('target_val', args.data_dir)

    test_loader_target = get_test_loader(dataset_target_val, args.height, args.width, args.batch_size, args.workers)

    # Create model
    source_classes = dataset_source.num_train_pids
    model = SDAModel(args, source_classes)      # create a model given args.model and other argsions

    # Evaluator
    evaluator_reid = Evaluator(model.net_B)
    _, mAP = evaluator_reid.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
    print('\n * Baseline mAP for target domain: {:5.1%}\n'.format(mAP))

    train_loader_source = get_train_loader(dataset_source, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters)
    train_loader_target = get_train_loader(dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, 0, iters)
    dataset_size = len(train_loader_source) * args.batch_size

    best_mAP_reid = best_mAP_reid_s = best_mAP_gan = 0

    for epoch in range(args.niter + args.niter_decay):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch

        train_loader_target.new_epoch()
        train_loader_source.new_epoch()

        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0

        model.set_status_init()

        for i in range(len(train_loader_source)):  # inner loop within one epoch
            source_inputs = train_loader_source.next()
            target_inputs = train_loader_target.next()

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % args.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_iters += args.batch_size
            epoch_iter += args.batch_size
            model.set_input(source_inputs, target_inputs)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch*len(train_loader_source)+i, epoch)   # calculate loss functions, get gradients, update network weights

            if total_iters % args.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % args.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % args.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / args.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if args.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % args.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if args.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % args.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')

        if ((epoch+1)%args.eval_step==0):

            _, mAP = evaluator_reid.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
            is_best = (mAP>best_mAP)
            best_mAP = max(mAP, best_mAP)
            print('\n * Target Domain: Finished epoch [{:3d}]  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(epoch, mAP, best_mAP, ' *' if is_best else ''))

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, args.niter + args.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.


if __name__ == '__main__':
    main()
