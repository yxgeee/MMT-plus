from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import shutil
import h5py
from tqdm import tqdm

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed

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

from visda.sda.options.test_options import TestOptions
from visda.sda.options.train_options import TrainOptions
from visda.sda.models.test_model import TestModel
from visda.sda.util.visualizer import Visualizer
from visda.sda.models import networks
from visda.sda.util.util import tensor2im, save_image

def get_data(name, data_dir):
    dataset = datasets.create(name, data_dir)
    return dataset

def get_test_loader(dataset, height, width, batch_size, workers):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    testset = sorted(dataset.train)

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def main():
    args = TrainOptions().parse()   # get training argsions

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    main_worker(args)

def main_worker(args):
    global start_epoch, best_mAP_reid, best_mAP_gan

    args.gpu = None
    args.rank = 0

    total_iters = 0                # the total number of training iterations

    cudnn.benchmark = True

    log_dir = osp.dirname(args.resume)
    print("==========\nArgs:{}\n==========".format(args))
    mkdir_if_missing(osp.join(log_dir, 'personX_sda', 'image_train'))

    # Create data loaders
    dataset_source = get_data('personx', args.data_dir)
    data_loader = get_test_loader(dataset_source, args.height, args.width, args.batch_size, args.workers)

    # Create model
    model = TestModel(args)      # create a model given args.model and other argsions
    model.load_networks('latest',args.resume)
    model.eval()


    # end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, pids, _, _) in enumerate(tqdm(data_loader)):

            model.set_input({'A':imgs, 'A_paths':fnames})
            model.test()
            visuals = model.get_current_visuals()  # get image results

            for fname, img_tensor in zip(fnames, visuals['fake']):
                img_np = tensor2im(img_tensor)
                save_image(img_np, osp.join(log_dir, 'personX_sda', 'image_train', osp.basename(fname)))


if __name__ == '__main__':
    main()
