from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import collections
import time
from datetime import timedelta

from sklearn.cluster import DBSCAN
import faiss
from apex import amp

import torch
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F

from visda import datasets
from visda import models
from visda.models.dsbn import convert_dsbn
from visda.models.moco import MoCo
from visda.trainers import MMTTrainer_UDA
from visda.evaluators import Evaluator, extract_features
from visda.utils.data import IterLoader
from visda.utils.data import transforms as T
from visda.utils.data.sampler import RandomMultipleGallerySampler
from visda.utils.data.preprocessor import Preprocessor
from visda.utils.logging import Logger
from visda.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from visda.utils.faiss_rerank import compute_jaccard_distance


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
                DataLoader(Preprocessor(train_set, root=dataset.images_dir,
                                        transform=train_transformer, mutual=True),
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

def create_model(args, num_classes):
    model_kwargs = {'num_features':args.features, 'dropout':args.dropout, 'num_classes':num_classes,
                    'metric':args.metric, 's':args.metric_s, 'm':args.metric_m}
    model_1 = models.create(args.arch, **model_kwargs)
    model_1_ema = models.create(args.arch, **model_kwargs)
    model_2 = models.create(args.arch, **model_kwargs)
    model_2_ema = models.create(args.arch, **model_kwargs)

    initial_weights = load_checkpoint(args.init_1)
    copy_state_dict(initial_weights['state_dict'], model_1, strip='module.')
    copy_state_dict(initial_weights['state_dict'], model_1_ema, strip='module.')
    model_1_ema.classifier.weight.data.copy_(model_1.classifier.weight.data)

    initial_weights = load_checkpoint(args.init_2)
    copy_state_dict(initial_weights['state_dict'], model_2, strip='module.')
    copy_state_dict(initial_weights['state_dict'], model_2_ema, strip='module.')
    model_2_ema.classifier.weight.data.copy_(model_2.classifier.weight.data)

    # adopt domain-specific BN
    convert_dsbn(model_1)
    convert_dsbn(model_2)
    convert_dsbn(model_1_ema)
    convert_dsbn(model_2_ema)

    # use CUDA
    model_1.cuda()
    model_2.cuda()
    model_1_ema.cuda()
    model_2_ema.cuda()

    # Optimizer
    optimizer = None
    if args.fp16:
        params = [{"params": [value]} for _, value in model_1.named_parameters() if value.requires_grad]
        params += [{"params": [value]} for _, value in model_2.named_parameters() if value.requires_grad]
        optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)
        # fp16
        [model_1, model_2], optimizer = amp.initialize([model_1, model_2], optimizer, opt_level="O1")

    # multi-gpu
    model_1 = nn.DataParallel(model_1)
    model_2 = nn.DataParallel(model_2)
    model_1_ema = nn.DataParallel(model_1_ema)
    model_2_ema = nn.DataParallel(model_2_ema)

    for param in model_1_ema.parameters():
        param.detach_()
    for param in model_2_ema.parameters():
        param.detach_()

    return model_1, model_2, model_1_ema, model_2_ema, optimizer


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
    start_time = time.monotonic()

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # Create datasets
    iters = args.iters if (args.iters>0) else None
    print("==> Load source-domain trainset")
    dataset_source = get_data(args.dataset_source, args.data_dir)
    print("==> Load target-domain trainset")
    dataset_target = get_data('target_train', args.data_dir)
    print("==> Load target-domain valset")
    dataset_target_val = get_data('target_val', args.data_dir)
    source_classes = dataset_source.num_train_pids

    # Create data loaders
    test_loader_target = get_test_loader(args, dataset_target_val, args.height, args.width, args.batch_size, args.workers)
    train_loader_source = get_train_loader(args, dataset_source, args.height, args.width,
                                        args.batch_size, args.workers, args.num_instances, iters, args.epochs)
    tgt_cluster_loader = get_test_loader(args, dataset_target, args.height, args.width, args.batch_size, args.workers,
                                        testset=sorted(dataset_target.train))

    # Create model
    all_classes = source_classes+len(dataset_target.train) if (args.cluster_alg=='dbscan') else source_classes+args.num_clusters
    model_1, model_2, model_1_ema, model_2_ema, optimizer = create_model(args, all_classes)

    # Evaluator
    evaluator_1_ema = Evaluator(model_1_ema)
    evaluator_2_ema = Evaluator(model_2_ema)
    evaluator_1 = Evaluator(model_1)
    evaluator_2 = Evaluator(model_2)

    evaluator_1_ema.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
    evaluator_2_ema.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)

    moco_1 = MoCo(dim=model_1.module.num_features, K=int(args.batch_size*iters*args.moco_neg), T=0.07).cuda()
    moco_2 = MoCo(dim=model_1.module.num_features, K=int(args.batch_size*iters*args.moco_neg), T=0.07).cuda()

    for epoch in range(args.epochs):

        dict_f, _ = extract_features(model_1_ema, tgt_cluster_loader, print_freq=50)
        cf_1 = torch.stack(list(dict_f.values()))
        dict_f, _ = extract_features(model_2_ema, tgt_cluster_loader, print_freq=50)
        cf_2 = torch.stack(list(dict_f.values()))
        target_features = (cf_1+cf_2)/2
        target_features = F.normalize(target_features, dim=1)

        if (args.cluster_alg=='dbscan'):
            # use jaccard distance instead of l2 distance
            jac_dist = compute_jaccard_distance(target_features.cuda(), k1=args.k1, k2=args.k2, search_option=3)

            if (epoch==0):
                # DBSCAN cluster
                print('Clustering criterion: eps: {:.3f}'.format(args.eps))
                cluster = DBSCAN(eps=args.eps, min_samples=args.min_samples, metric='precomputed', n_jobs=-1)

            print('Clustering and labeling...')
            labels = cluster.fit_predict(jac_dist)
            num_ids = len(set(labels)) - (1 if -1 in labels else 0)
            args.num_clusters = num_ids
            print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'
                        .format(epoch, args.num_clusters, (labels==-1).sum()))
            del jac_dist

            # generate new dataset and calculate cluster centers
            new_dataset = []
            outliers = 0
            cluster_centers = collections.defaultdict(list)
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), labels)):
                if label==-1:
                    continue

                new_dataset.append((fname, source_classes+label, cid))
                cluster_centers[label].append(target_features[i])

            cluster_centers = [torch.stack(cluster_centers[idx]).mean(0) for idx in sorted(cluster_centers.keys())]
            cluster_centers = F.normalize(torch.stack(cluster_centers)).float().cuda()

            args.num_clusters += outliers

        elif (args.cluster_alg=='kmeans'):

            if (epoch==0):
                # k-means cluster by faiss
                cluster = faiss.Kmeans(target_features.size(-1), args.num_clusters,
                                        niter=300, verbose=True, gpu=True)

            cluster.train(target_features.numpy())
            cluster_centers = F.normalize(torch.from_numpy(cluster.centroids)).float().cuda()
            _, labels = cluster.index.search(target_features.numpy(),1)
            # generate new dataset
            new_dataset = []
            for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset_target.train), labels)):
                if label[0]==-1: continue
                new_dataset.append((fname, source_classes+label[0], cid))

        else:
            assert("unknown cluster algorithm {}, please use dbscan or kmeans".format(args.cluster_alg))

        model_1.module.classifier.weight.data[source_classes:source_classes+args.num_clusters].copy_(cluster_centers)
        model_2.module.classifier.weight.data[source_classes:source_classes+args.num_clusters].copy_(cluster_centers)
        model_1_ema.module.classifier.weight.data[source_classes:source_classes+args.num_clusters].copy_(cluster_centers)
        model_2_ema.module.classifier.weight.data[source_classes:source_classes+args.num_clusters].copy_(cluster_centers)

        del target_features, cluster_centers

        train_loader_target = get_train_loader(args, dataset_target, args.height, args.width,
                                            args.batch_size, args.workers, args.num_instances, iters, args.epochs,
                                            trainset=new_dataset)

        if (not args.fp16):
            # Optimizer
            params = [{"params": [value]} for _, value in model_1.named_parameters() if value.requires_grad]
            params += [{"params": [value]} for _, value in model_2.named_parameters() if value.requires_grad]
            optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.weight_decay)

        # Trainer
        trainer = MMTTrainer_UDA(model_1, model_2, model_1_ema, model_2_ema, moco_1, moco_2,
                                source_classes+args.num_clusters, margin=args.margin, alpha=args.alpha)

        train_loader_source.new_epoch()
        train_loader_target.new_epoch()

        trainer.train(epoch, train_loader_source, train_loader_target, optimizer, fp16=args.fp16,
                    ce_soft_weight=args.soft_ce_weight, tri_soft_weight=args.soft_tri_weight,
                    tri_weight=args.tri_weight, mc_weight=args.mc_weight,
                    print_freq=args.print_freq, train_iters=len(train_loader_target))

        def save_model(model, is_best, best_mAP, mid):
            save_checkpoint({
                'state_dict': model.state_dict(),
                'epoch': epoch + 1,
                'best_mAP': best_mAP,
            }, is_best, fpath=osp.join(args.logs_dir, 'model'+str(mid)+'_checkpoint.pth.tar'))

        if ((epoch+1)%args.eval_step==0 or (epoch==args.epochs-1)):
            _, mAP_1s = evaluator_1.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
            _, mAP_2s = evaluator_2.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
            _, mAP_1 = evaluator_1_ema.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
            _, mAP_2 = evaluator_2_ema.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)
            is_best = (mAP_1>best_mAP) or (mAP_2>best_mAP)
            best_mAP = max(mAP_1, mAP_2, best_mAP)
            save_model(model_1_ema, (is_best and (mAP_1>mAP_2)), best_mAP, 1)
            save_model(model_2_ema, (is_best and (mAP_1<=mAP_2)), best_mAP, 2)

            print('\n * Finished epoch {:3d}  model no.1 mAP: {:5.1%} ({:5.1%}) model no.2 mAP: {:5.1%} ({:5.1%})  best: {:5.1%}{}\n'.
                  format(epoch, mAP_1, mAP_1s, mAP_2, mAP_2s, best_mAP, ' *' if is_best else ''))

    print ('Test on the best model.')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model_1_ema.load_state_dict(checkpoint['state_dict'])
    evaluator_1_ema.evaluate(test_loader_target, dataset_target_val.query, dataset_target_val.gallery)

    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMT+ Training")
    # data
    parser.add_argument('-ds', '--dataset-source', type=str, default='personx',
                        choices=datasets.names())
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--height', type=int, default=384,
                        help="input height")
    parser.add_argument('--width', type=int, default=128,
                        help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    # cluster
    parser.add_argument('--cluster-alg', type=str, default='dbscan',
                        help="dbscan or kmeans")
    parser.add_argument('--eps', type=float, default=0.6,
                        help="max neighbor distance for DBSCAN")
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--num-clusters', type=int, default=0,
                        help="hyperparameter for k-means")
    parser.add_argument('--min-samples', type=int, default=4,
                        help="hyperparameter for DBSCAN")
    parser.add_argument('--camera', type=str, default='', metavar='PATH')
    # model
    parser.add_argument('-a', '--arch', type=str, default='resnet_ibn50a',
                        choices=models.names())
    parser.add_argument('-ac', '--arch-c', type=str, default='resnet_ibn50a',
                        choices=models.names())
    parser.add_argument('-m', '--metric', type=str, default='cos')
    parser.add_argument('-ms', '--metric-s', type=float, default=64)
    parser.add_argument('-mm', '--metric-m', type=float, default=0.35)
    parser.add_argument('--features', type=int, default=0)
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--moco-neg', type=float, default=1)
    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035,
                        help="learning rate of new parameters, for pretrained "
                             "parameters it is 10 times smaller than this")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)
    # training configs
    parser.add_argument('--init-1', type=str, default='', metavar='PATH')
    parser.add_argument('--init-2', type=str, default='', metavar='PATH')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=1)
    parser.add_argument('--margin', type=float, default=0.0, help='margin for the triplet loss with batch hard')
    parser.add_argument('--soft-ce-weight', type=float, default=0.5)
    parser.add_argument('--soft-tri-weight', type=float, default=0.5)
    parser.add_argument('--tri-weight', type=float, default=0)
    parser.add_argument('--mc-weight', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.999)
    parser.add_argument('--fp16', action='store_true', help="training only")
    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    main()
