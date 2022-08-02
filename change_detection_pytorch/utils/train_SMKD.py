import os
import os.path as osp
import sys
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from tqdm import tqdm as tqdm

from .meter import AverageValueMeter


class Epoch:

    def __init__(self, args, model, teacher, loss, metrics, stage_name, device='cpu', verbose=True):
        self.args = args
        self.model = model
        self.teacher = teacher
        self.bestmodel = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        if self.teacher is not None:
            self.teacher.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def save_bestmodel(self):
        self.bestmodel = copy.deepcopy(self.model)

    def batch_update(self, x1, x2, y, epoch):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def check_tensor(self, data, is_label):
        if not is_label:
            return data if data.ndim <= 4 else data.squeeze()
        return data.long() if data.ndim <= 3 else data.squeeze().long()

    def infer_vis(self, dataloader, save=True, evaluate=False, slide=False, image_size=512,
                  window_size=256, save_dir='./res', suffix='.tif'):
        """
        Infer and save results. (debugging)
        Note: Currently only batch_size=1 is supported.
        Weakly robust.
        'image_size' and 'window_size' work when slide is True.
        """
        import cv2
        import numpy as np

        self.model.eval()
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, mode=0o777)
        logs = {}
        # AverageValueMeter class is a auto-mean tool
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for (x1, x2, y, filename) in iterator:

                assert y is not None or not evaluate, "When the label is None, the evaluation mode cannot be turned on."

                if y is not None:
                    x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), self.check_tensor(y, True)
                    x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                else:
                    x1, x2 = self.check_tensor(x1, False), self.check_tensor(x2, False)
                    x1, x2 = x1.float(), x2.float()
                    x1, x2 = x1.to(self.device), x2.to(self.device)
                y_pred, _ = self.bestmodel.forward(x1, x2)

                if evaluate:
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

                if save:
                    y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
                    y_pred = y_pred * 255
                    filename = filename[0].split('.')[0] + suffix

                    if slide:
                        inf_seg_maps = []
                        window_num = image_size // window_size
                        window_idx = [i for i in range(0, window_num ** 2 + 1, window_num)]
                        for row_idx in range(len(window_idx) - 1):
                            inf_seg_maps.append(np.concatenate([y_pred[i] for i in range(window_idx[row_idx],
                                                                                         window_idx[row_idx + 1])],
                                                               axis=1))
                        inf_seg_maps = np.concatenate([row for row in inf_seg_maps], axis=0)
                        cv2.imwrite(osp.join(save_dir, filename), inf_seg_maps)
                    else:
                        # To be verified
                        cv2.imwrite(osp.join(save_dir, filename), y_pred)

    def run(self, dataloader, epoch):
        # train mode switch
        self.on_epoch_start()

        logs = {}
        loss_meter = {'loss': AverageValueMeter()}
        if self.stage_name == 'train' and self.args.model_name == 'SMKD':
            loss_meter = {'loss': AverageValueMeter(), 'stuloss': AverageValueMeter(), 'teacloss': AverageValueMeter(),
                          'kld': AverageValueMeter(), 'dis': AverageValueMeter()}
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for (x1, x2, y, filename) in iterator:

                x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                            self.check_tensor(y, True)
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x1, x2, y, epoch)

                # update loss logs
                for k, v in loss.items():
                    loss_value = v.detach().cpu().numpy()
                    loss_meter[k].add(loss_value)
                    loss_logs = {k: loss_meter[k].mean}
                    logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, args, model, teacher, loss, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            args=args,
            model=model,
            teacher=teacher,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer
        self.linear = DataParallel(torch.nn.Linear(2048, 512).to(device),
                                   device_ids=[k for k in range(args.num_device)])

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x1, x2, y, epoch):
        self.optimizer.zero_grad()

        # if epoch == self.args.warmup_epoch:
        #     for item in self.teacher.parameters():
        #         item.requires_grad = False

        prediction, feature = self.model.forward(x1, x2)
        student_loss = self.loss(prediction, y)

        if self.teacher is not None:
            batch, h = x1.shape[0], x1.shape[-1]

            prediction_teacher, feature_teacher = self.teacher.forward(x1, x2)
            teacher_loss = self.args.alpha * self.loss(prediction_teacher, y)

            kl_loss = nn.KLDivLoss(reduction='batchmean')
            kld = self.args.beta * kl_loss(F.log_softmax(prediction, dim=1),
                                           F.softmax(prediction_teacher, dim=1)) / h / h

            dis_loss = nn.PairwiseDistance(p=self.args.norm)
            stu_1 = feature[0][-1].permute(0, 2, 3, 1).reshape(batch, -1)
            stu_2 = feature[1][-1].permute(0, 2, 3, 1).reshape(batch, -1)
            tea_1 = self.linear(feature_teacher[0][-1].permute(0, 2, 3, 1)).reshape(batch, -1)
            tea_2 = self.linear(feature_teacher[1][-1].permute(0, 2, 3, 1)).reshape(batch, -1)
            dis = dis_loss(stu_1, tea_1) + dis_loss(stu_2, tea_2)
            dis = self.args.gamma * torch.mean(dis) / h / h
            loss = student_loss + teacher_loss + kld + dis

            loss.backward()
            self.optimizer.step()
            return {'loss': loss, 'stuloss': student_loss, 'teacloss': teacher_loss, 'kld': kld, 'dis': dis}, prediction
        else:
            loss = student_loss
            loss.backward()
            self.optimizer.step()
            return {'loss': loss}, prediction


class ValidEpoch(Epoch):

    def __init__(self, args, model, teacher, loss, metrics, device='cpu', verbose=True):
        super().__init__(
            args=args,
            model=model,
            teacher=teacher,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x1, x2, y, epoch):
        with torch.no_grad():
            prediction, _ = self.model.forward(x1, x2)
            loss = self.loss(prediction, y)
        return {'loss': loss}, prediction
