from __future__ import absolute_import

from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import torch

from .gem import GeneralizedMeanPoolingP
from .metric import build_metric

from .densenet import densenet121_ibn_a, densenet169_ibn_a


__all__ = ['densenet_ibn121a', 'densenet_ibn169a']


class DenseNet_IBN(nn.Module):
    __factory = {
        '121a': densenet121_ibn_a,
        '169a': densenet169_ibn_a
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0,
                 metric=None, s=64, m=0.35):
        super(DenseNet_IBN, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling

        resnet = self.__factory[depth](pretrained=True)

        self.base = nn.Sequential(resnet.features, nn.ReLU())
        self.gap = GeneralizedMeanPoolingP()

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes

            out_planes = resnet.classifier.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
                self.feat_bn = nn.BatchNorm1d(self.num_features)
            self.feat_bn.bias.requires_grad_(False)
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                if metric is not None:
                    self.classifier = build_metric(metric, self.num_features, self.num_classes, s, m)
                else:
                    self.classifier = nn.Linear(self.num_features, self.num_classes, bias=False)
                    init.normal_(self.classifier.weight, std=0.001)

        init.constant_(self.feat_bn.weight, 1)
        init.constant_(self.feat_bn.bias, 0)

        if not pretrained:
            self.reset_params()

    def forward(self, x, y=None):
        x = self.base(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        if self.cut_at_pooling:
            return x

        if self.has_embedding:
            bn_x = self.feat_bn(self.feat(x))
        else:
            bn_x = self.feat_bn(x)

        if self.training is False:
            return bn_x

        if self.norm:
            bn_x = F.normalize(bn_x)
        elif self.has_embedding:
            bn_x = F.relu(bn_x)

        if self.dropout > 0:
            bn_x = self.drop(bn_x)

        if self.num_classes > 0:
            if isinstance(self.classifier, nn.Linear):
                prob = self.classifier(bn_x)
            else:
                prob = self.classifier(bn_x, y)
        else:
            return bn_x

        return x, bn_x, prob

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


def densenet_ibn121a(**kwargs):
    return DenseNet_IBN('121a', **kwargs)

def densenet_ibn169a(**kwargs):
    return DenseNet_IBN('169a', **kwargs)
