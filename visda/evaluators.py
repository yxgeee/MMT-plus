from __future__ import print_function, absolute_import
import time
import collections
from collections import OrderedDict
import numpy as np
import torch
import random
import copy
import collections

from .evaluation_metrics import eval_func
from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch

def extract_cnn_feature(model, inputs, inputs_flip=None):
    inputs = to_torch(inputs).cuda()
    outputs = model(inputs)
    if inputs_flip is not None:
        outputs_flip = model(inputs_flip)
        outputs += outputs_flip
        outputs /= 2
    outputs = torch.nn.functional.normalize(outputs)
    return outputs

def extract_features(model, data_loader, print_freq=50, flip=False):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = OrderedDict()
    labels = OrderedDict()

    end = time.time()
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if flip:
                imgs = data[0]
                imgs_flip = data[1]
                fnames = data[2]
                pids = data[3]
            else:
                imgs = data[0]
                fnames = data[1]
                pids = data[2]

            data_time.update(time.time() - end)

            if flip:
                outputs = extract_cnn_feature(model, imgs, imgs_flip)
            else:
                outputs = extract_cnn_feature(model, imgs)

            outputs = outputs.data.cpu()

            for fname, output, pid in zip(fnames, outputs, pids):
                features[fname] = output
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    return features, labels


def pairwise_distance(features, query=None, gallery=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m

    x = torch.cat([features[f].unsqueeze(0) for f, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _ in gallery], 0)
    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x, y


def evaluate_all(distmat, query=None, gallery=None,
                 query_ids=None, gallery_ids=None,
                 query_cams=None, gallery_cams=None,
                 cmc_topk=(1, 5, 10, 20, 50),
                 submit_file=None, only_submit=False):

    if (not only_submit):
        if query is not None and gallery is not None:
            query_ids = [pid for _, pid, _ in query]
            gallery_ids = [pid for _, pid, _ in gallery]
            query_cams = [cam for _, _, cam in query]
            gallery_cams = [cam for _, _, cam in gallery]
        else:
            assert (query_ids is not None and gallery_ids is not None
                    and query_cams is not None and gallery_cams is not None)

        cmc_scores, mAP = eval_func(distmat, np.array(query_ids), np.array(gallery_ids),
                            np.array(query_cams), np.array(gallery_cams), max_rank=50, ap_topk=100)
        print('Mean AP: {:4.1%}'.format(mAP))
        print('CMC Scores:')
        for k in cmc_topk:
            print('  top-{:<4}{:12.1%}'.format(k, cmc_scores[k-1]))
    else:
        cmc_scores, mAP = [], 0

    if submit_file:
        indices = np.argsort(distmat, axis=1)
        np.savetxt(submit_file, indices[:, :100], fmt="%05d")

    return cmc_scores, mAP


class Evaluator(object):
    def __init__(self, model, cam_model=None, cam_weight=0.1, flip=False):
        super(Evaluator, self).__init__()
        self.model = model
        self.cam_model = cam_model
        self.cam_weight = cam_weight
        self.flip = flip

    def cam_dist(self, cam_features, query, gallery):
        if (cam_features is None):
            return 0
        cam_distmat, _, _ = pairwise_distance(cam_features, query, gallery)
        return self.cam_weight*cam_distmat

    def evaluate(self, data_loader, query, gallery, features=None,
                    rerank=False, k1=20, k2=6, lambda_value=0.3,
                    submit_file=None, qe=False, only_submit=False):
        if (features is None):
            features, _ = extract_features(self.model, data_loader, flip=self.flip)

        # cam_features = None
        if (self.cam_model):
            cam_features, _ = extract_features(self.cam_model, data_loader, flip=self.flip)
        else:
            cam_features = None

        distmat, _, _ = pairwise_distance(features, query, gallery)
        distmat -= self.cam_dist(cam_features, query, gallery)
        results = evaluate_all(distmat.numpy(), query=query, gallery=gallery,
                    submit_file=(None if rerank else submit_file), only_submit=only_submit)

        if (not rerank):
            return results

        print('Applying person re-ranking ...')
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)

        distmat_qq -= self.cam_dist(cam_features, query, query)
        distmat_gg -= self.cam_dist(cam_features, gallery, gallery)

        distmat_rr = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                                k1=k1, k2=k2, lambda_value=lambda_value)
        results = evaluate_all(distmat_rr, query=query, gallery=gallery,
                    submit_file=submit_file, only_submit=only_submit)

        return results
