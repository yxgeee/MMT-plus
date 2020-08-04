from __future__ import print_function, absolute_import
import time
import numpy as np
import collections

from apex import amp

import torch
import torch.nn as nn
from torch.nn import functional as F

from .evaluation_metrics import accuracy
from .loss import TripletLoss, CrossEntropyLabelSmooth, SoftTripletLoss, SoftEntropy
from .utils.meters import AverageMeter


class PreTrainer(object):
    def __init__(self, model, num_classes, margin=0.0, fp16=False):
        super(PreTrainer, self).__init__()
        self.model = model
        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_triple = SoftTripletLoss(margin=margin).cuda()
        self.fp16 = fp16

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, train_iters=200, print_freq=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        losses_tr = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            s_inputs, targets = self._parse_data(source_inputs)
            t_inputs, _ = self._parse_data(target_inputs)
            s_features, _, s_cls_out = self.model(s_inputs)
            # target samples: only forward
            t_features, _, _ = self.model(t_inputs)

            # backward main #
            loss_ce, loss_tr, prec1 = self._forward(s_features, s_cls_out, targets)
            loss = loss_ce + loss_tr

            losses_ce.update(loss_ce.item())
            losses_tr.update(loss_tr.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            if self.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            # loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tr {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tr.val, losses_tr.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        inputs = imgs.cuda()
        targets = pids.cuda()
        return inputs, targets

    def _forward(self, s_features, s_outputs, targets):
        loss_ce = self.criterion_ce(s_outputs, targets)
        if isinstance(self.criterion_triple, SoftTripletLoss):
            loss_tr = self.criterion_triple(s_features, s_features, targets)
        elif isinstance(self.criterion_triple, TripletLoss):
            loss_tr, _ = self.criterion_triple(s_features, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, loss_tr, prec



class MMTTrainer_UDA(object):
    def __init__(self, model_1, model_2, model_1_ema, model_2_ema, moco_1, moco_2,
                    num_classes, margin=0.0, alpha=0.999):
        super(MMTTrainer_UDA, self).__init__()
        self.model_1 = model_1
        self.model_2 = model_2
        self.model_1_ema = model_1_ema
        self.model_2_ema = model_2_ema
        self.moco_1 = moco_1
        self.moco_2 = moco_2

        self.num_classes = num_classes
        self.alpha = alpha

        self.criterion_ce = CrossEntropyLabelSmooth(num_classes).cuda()
        self.criterion_ce_soft = SoftEntropy().cuda()
        self.criterion_tri = SoftTripletLoss(margin=margin).cuda()
        self.criterion_tri_soft = SoftTripletLoss(margin=None).cuda()

    def train(self, epoch, data_loader_source, data_loader_target, optimizer, fp16=False,
             ce_soft_weight=0.5, tri_soft_weight=0.5, tri_weight=1.0, mc_weight=0,
             print_freq=10, train_iters=400):

        self.model_1.train()
        self.model_2.train()
        self.model_1_ema.train()
        self.model_2_ema.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()

        losses_ce = AverageMeter()
        losses_tri = AverageMeter()
        losses_ce_soft = AverageMeter()
        losses_tri_soft = AverageMeter()
        losses_mc = AverageMeter()
        precisions = [AverageMeter(),AverageMeter()]

        end = time.time()
        for i in range(train_iters):
            # load data
            source_inputs = data_loader_source.next()
            target_inputs = data_loader_target.next()
            data_time.update(time.time() - end)

            # process inputs
            s_inputs_1, s_inputs_2, s_targets = self._parse_data(source_inputs)
            t_inputs_1, t_inputs_2, t_targets = self._parse_data(target_inputs)

            # rearrange batch for domain-specific BN
            device_num = torch.cuda.device_count()
            B, C, H, W = s_inputs_1.size()
            def reshape(inputs):
                return inputs.view(device_num, -1, C, H, W)

            s_inputs_1, s_inputs_2, t_inputs_1, t_inputs_2 = reshape(s_inputs_1), reshape(s_inputs_2), \
                                                            reshape(t_inputs_1), reshape(t_inputs_2)
            inputs_1, inputs_2 = torch.cat((s_inputs_1, t_inputs_1), 1), torch.cat((s_inputs_2, t_inputs_2), 1)
            inputs_1, inputs_2 = inputs_1.view(-1, C, H, W), inputs_2.view(-1, C, H, W)

            s_targets, t_targets = s_targets.view(device_num, -1), t_targets.view(device_num, -1)
            targets = torch.cat((s_targets, t_targets), 1)
            targets = targets.view(-1)

            # forward
            f_out_1, p_out_1, f_out_2, p_out_2, \
                    f_out_1_ema, p_out_1_ema, f_out_2_ema, p_out_2_ema,\
                        q1, q2, k1, k2 = self._forward(inputs_1, inputs_2, targets)

            # de-arrange batch
            def debatch(fea):
                fea = fea.view(device_num, -1, fea.size(-1))
                fea_s, fea_t = fea.split(fea.size(1)//2, dim=1)
                fea_s, fea_t = fea_s.contiguous().view(-1, fea.size(-1)), fea_t.contiguous().view(-1, fea.size(-1))
                return fea_s, fea_t

            _, q1_t = debatch(q1)
            _, q2_t = debatch(q2)
            _, k1_t = debatch(k1)
            _, k2_t = debatch(k2)

            # compute loss
            loss_ce = self.criterion_ce(p_out_1, targets) + self.criterion_ce(p_out_2, targets)
            loss_tri = self.criterion_tri(f_out_1, f_out_1, targets) + self.criterion_tri(f_out_2, f_out_2, targets)

            loss_ce_soft = self.criterion_ce_soft(p_out_1, p_out_2_ema) + self.criterion_ce_soft(p_out_2, p_out_1_ema)
            loss_tri_soft = self.criterion_tri_soft(f_out_1, f_out_2_ema, targets) + \
                            self.criterion_tri_soft(f_out_2, f_out_1_ema, targets)

            loss_mc = self.moco_1(q1_t, k1_t)+self.moco_2(q2_t, k2_t)

            loss = loss_ce*(1-ce_soft_weight) + loss_ce_soft*ce_soft_weight + \
                    (loss_tri*(1-tri_soft_weight) + loss_tri_soft*tri_soft_weight)*tri_weight + \
                    loss_mc * mc_weight

            # backpropagate
            optimizer.zero_grad()
            if fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()
            optimizer.step()

            self._update_ema_variables(self.model_1, self.model_1_ema, self.alpha, epoch*len(data_loader_target)+i)
            self._update_ema_variables(self.model_2, self.model_2_ema, self.alpha, epoch*len(data_loader_target)+i)

            prec_1, = accuracy(p_out_1.data, targets.data)
            prec_2, = accuracy(p_out_2.data, targets.data)

            losses_ce.update(loss_ce.item())
            losses_tri.update(loss_tri.item())
            losses_ce_soft.update(loss_ce_soft.item())
            losses_tri_soft.update(loss_tri_soft.item())
            losses_mc.update(loss_mc.item())
            precisions[0].update(prec_1[0])
            precisions[1].update(prec_2[0])

            # print log #
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Loss_tri {:.3f} ({:.3f})\t'
                      'Loss_ce_soft {:.3f} ({:.3f})\t'
                      'Loss_tri_soft {:.3f} ({:.3f})\t'
                      'Loss_mc {:.3f} ({:.3f})\t'
                      'Prec {:.2%} / {:.2%}'
                      .format(epoch, i + 1, len(data_loader_target),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              losses_tri.val, losses_tri.avg,
                              losses_ce_soft.val, losses_ce_soft.avg,
                              losses_ce_soft.val, losses_tri_soft.avg,
                              losses_mc.val, losses_mc.avg,
                              precisions[0].avg, precisions[1].avg))

    def _update_ema_variables(self, model, ema_model, alpha, global_step):
        alpha = min(1 - 1 / (global_step + 1), alpha)
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

    def _parse_data(self, inputs):
        imgs_1, imgs_2, pids, _ = inputs
        return imgs_1.cuda(), imgs_2.cuda(), pids.cuda()

    def _forward(self, inputs_1, inputs_2, targets):
        f_out_t1, q1, p_out_t1 = self.model_1(inputs_1, targets)
        f_out_t2, q2, p_out_t2 = self.model_2(inputs_2, targets)

        with torch.no_grad():
            f_out_t1_ema, k1, p_out_t1_ema = self.model_1_ema(inputs_1, targets)
            f_out_t2_ema, k2, p_out_t2_ema = self.model_2_ema(inputs_2, targets)

        return f_out_t1, p_out_t1[:,:self.num_classes], f_out_t2, p_out_t2[:,:self.num_classes], \
                f_out_t1_ema, p_out_t1_ema[:,:self.num_classes], f_out_t2_ema, p_out_t2_ema[:,:self.num_classes], \
                    q1, q2, k1, k2


class CameraTrainer(object):
    def __init__(self, model, num_classes, margin=0.0, fp16=False):
        super(CameraTrainer, self).__init__()
        self.model = model
        self.fp16 = fp16

    def train(self, epoch, data_loader, optimizer, train_iters=200, print_freq=10):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_ce = AverageMeter()
        precisions = AverageMeter()

        end = time.time()

        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            inputs, targets = self._parse_data(inputs)
            _, _, cls_out = self.model(inputs, targets)

            # backward main #
            loss_ce, prec1 = self._forward(cls_out, targets)
            # loss = loss_ce

            losses_ce.update(loss_ce.item())
            precisions.update(prec1)

            optimizer.zero_grad()
            if self.fp16:
                with amp.scale_loss(loss_ce, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss_ce.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_ce {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})'
                      .format(epoch, i + 1, train_iters,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_ce.val, losses_ce.avg,
                              precisions.val, precisions.avg))

    def _parse_data(self, inputs):
        imgs, _, _, camids, _ = inputs
        inputs = imgs.cuda()
        targets = camids.cuda()
        return inputs, targets

    def _forward(self, s_outputs, targets):
        loss_ce = F.cross_entropy(s_outputs, targets)
        prec, = accuracy(s_outputs.data, targets.data)
        prec = prec[0]

        return loss_ce, prec
