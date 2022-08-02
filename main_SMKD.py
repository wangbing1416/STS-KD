import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn import DataParallel
import numpy as np
import random
import os
import sys
import logging
import uuid
from time import strftime, localtime

import change_detection_pytorch as cdp
import change_detection_pytorch.utils.train_SMKD as Traniner
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler
import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True


class Instructor:
    def __init__(self, args):
        self.args = args
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = args.modelClass(
            encoder_name=args.encoder,
            encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.args.classes,
            siam_encoder=True,  # whether to use a siamese encoder
            fusion_form=self.args.fusion,
        )
        self.model = DataParallel(self.model, device_ids=[k for k in range(args.num_device)])
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.args.lr,
                                          weight_decay=self.args.weight_decay)
        self.teacher = None
        if self.args.model_name == 'SMKD':
            self.teacher = args.modelClass(
                encoder_name=args.teacher,
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.args.classes,
                siam_encoder=True,  # whether to use a siamese encoder
                fusion_form=self.args.fusion,
            )
            self.teacher = DataParallel(self.teacher, device_ids=[k for k in range(args.num_device)])
            self.optimizer = torch.optim.Adam(params=[{'params': self.model.parameters()},
                                                      {'params': self.teacher.parameters()}],
                                              lr=self.args.lr, weight_decay=self.args.weight_decay)

        self.train_dataset = args.datasetClass(self.args.train_dir,
                                               sub_dir_1='A', sub_dir_2='B',
                                               img_suffix='.png',
                                               ann_dir=args.train_dir + '/label',
                                               debug=False)
        self.valid_dataset = args.datasetClass(self.args.val_dir,
                                               sub_dir_1='A', sub_dir_2='B',
                                               img_suffix='.png',
                                               ann_dir=args.val_dir + '/label',
                                               debug=False, test_mode=True)
        self.test_dataset = args.datasetClass(self.args.test_dir,
                                              sub_dir_1='A', sub_dir_2='B',
                                              img_suffix='.png',
                                              ann_dir=args.test_dir + '/label',
                                              debug=False, test_mode=True)

        self.train_loader = DataLoader(self.train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=1, shuffle=False, num_workers=1)
        self.test_loader = DataLoader(self.test_dataset, batch_size=1, shuffle=False, num_workers=1)

        self.loss = cdp.utils.losses.CrossEntropyLoss()
        self.metrics = [
            cdp.utils.metrics.Fscore(activation='argmax2d'),
            cdp.utils.metrics.Precision(activation='argmax2d'),
            cdp.utils.metrics.Recall(activation='argmax2d'),
            cdp.utils.metrics.IoU(activation='argmax2d'),
            # cdp.utils.metrics.Kappa(activation='argmax2d'),
        ]

        self.scheduler_steplr = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, ], gamma=0.1)

        # create epoch runners
        # it is a simple loop of iterating over dataloader`s samples
        self.train_epoch = Traniner.TrainEpoch(
            self.args,
            self.model,
            self.teacher,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=DEVICE,
            verbose=True,
        )

        self.valid_epoch = Traniner.ValidEpoch(
            self.args,
            self.model,
            self.teacher,
            loss=self.loss,
            metrics=self.metrics,
            device=DEVICE,
            verbose=True,
        )

        if self.args.num_device >= 1:
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated()))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info(
            'n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.args):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.args, arg)))

    def run(self):
        # train model for 60 epochs
        save_path = './checkpoint/best_model-{}.pth'.format(self.args.id)
        max_score = {'fscore': 0, 'precision': 0, 'recall': 0, 'iou_score': 0}
        for i in range(self.args.epoch):
            logger.info('\nEpoch: {}'.format(i))
            train_logs = self.train_epoch.run(self.train_loader, i)
            valid_logs = self.valid_epoch.run(self.valid_loader, i)
            self.scheduler_steplr.step()
            if self.args.model_name == 'SMKD':
                logger.info('train -> F1: {:.4f}, P: {:.4f}, R: {:.4f}, IOU: {:.4f}, loss: {:.4f}, stuloss: {:.4f},'
                            'teacloss: {:.4f}, kld: {:.4f}, dis: {:.4f}'.
                            format(train_logs['fscore'], train_logs['precision'], train_logs['recall'],
                                   train_logs['iou_score'], train_logs['loss'], train_logs['stuloss'],
                                   train_logs['teacloss'], train_logs['kld'], train_logs['dis']))
            else:
                logger.info('train -> F1: {:.4f}, P: {:.4f}, R: {:.4f}, IOU: {:.4f}, loss: {:.4f}'.
                            format(train_logs['fscore'], train_logs['precision'],
                                   train_logs['recall'], train_logs['iou_score'], train_logs['loss']))
            logger.info('eval -> F1: {:.4f}, P: {:.4f}, R: {:.4f}, IOU: {:.4f}, loss: {:.4f}'.
                        format(valid_logs['fscore'], valid_logs['precision'],
                               valid_logs['recall'], valid_logs['iou_score'], valid_logs['loss']))
            # do something (save model, change lr, etc.)
            if max_score['fscore'] < valid_logs['fscore']:
                max_score['fscore'] = valid_logs['fscore']
                self.train_epoch.save_bestmodel()  # record best model for inference
                torch.save(self.model, save_path)
                logger.info('Model has been saved in ' + save_path)

        logger.info(''.join(['>' for _ in range(20)]))
        test_logs = self.valid_epoch.run(self.test_loader, 0)
        logger.info('Final Test Result:')
        logger.info('F1: {:.4f}, P: {:.4f}, R: {:.4f}, IOU: {:.4f}'.
                    format(test_logs['fscore'], test_logs['precision'],
                           test_logs['recall'], test_logs['iou_score']))

        # save results (change maps)
        test_path = './log/{}/res-{}'.format(self.args.model_name, self.args.id)
        # inference and save result
        self.valid_epoch.infer_vis(self.test_loader, save=True, slide=False, save_dir=test_path)


def main():
    dataset2dir = {
        'LEVIR': {
            'train': './data/LEVIR-CD/train',
            'val': './data/LEVIR-CD/val',
            'test': './data/LEVIR-CD/test'
        },
        'WHU': {
            'train': './data/WHU-CD/train',
            'val': './data/WHU-CD/val',
            'test': './data/WHU-CD/test',
        }
    }
    dataset2class = {
        'LEVIR': LEVIR_CD_Dataset,
        'WHU': LEVIR_CD_Dataset,
    }
    model2class = {
        'UNet': cdp.Unet,
        'DeepLabV3+': cdp.deeplabv3.DeepLabV3Plus,
        'FPN': cdp.fpn.FPN,
        'PSPNet': cdp.pspnet.PSPNet,
        'UNet++': cdp.unetplusplus.UnetPlusPlus,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='SMKD', type=str, help='Baseline, SMKD')
    parser.add_argument('--model', default='FPN', type=str, help='UNet, DeepLabV3+, FPN, PSPNet')
    parser.add_argument('--encoder', default='resnet34', type=str,
                        help='choose encoder, e.g. resnet18, resnet34, resnet50, resnet101, resnet152, '
                             'vgg19, mobilenet_v2, Swin-T')
    parser.add_argument('--dataset', default='WHU', type=str, help='LEVIR, WHU')

    parser.add_argument('--fusion', default='concat', type=str,
                        help='the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.')
    parser.add_argument('--classes', default=2, type=int,
                        help='model output channels (number of classes in your datasets)')

    parser.add_argument('--teacher', default='resnet101', type=str)
    # parser.add_argument('--warmup_epoch', default=40, type=int)
    parser.add_argument('--alpha', default=1.0, type=float)
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--gamma', default=1.0, type=float)
    parser.add_argument('--norm', default=2, type=int)

    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--weight_decay', default=0.00000, type=float)
    parser.add_argument('--epoch', default=60, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--num_device', default=2, type=int, help='the number of training device')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    args.id = str(uuid.uuid4())[:8]
    args.train_dir = dataset2dir[args.dataset]['train']
    args.val_dir = dataset2dir[args.dataset]['val']
    args.test_dir = dataset2dir[args.dataset]['test']
    args.modelClass = model2class[args.model]
    args.datasetClass = dataset2class[args.dataset]
    if args.model_name == 'Baseline':
        args.warmup_epoch = 0
    setup_seed(args.seed)

    if not os.path.exists('./log/{}'.format(args.model_name)):
        os.makedirs('./log/{}'.format(args.model_name), mode=0o777)
    log_file = '{}-{}-{}-{}-{}-{}.log'.format(args.model_name, args.model, args.encoder,
                                              args.dataset, strftime("%Y-%m-%d_%H:%M:%S", localtime()), args.id)
    logger.addHandler(logging.FileHandler("%s/%s" % ('./log/{}'.format(args.model_name), log_file)))

    ins = Instructor(args)
    ins.run()


if __name__ == '__main__':
    main()

