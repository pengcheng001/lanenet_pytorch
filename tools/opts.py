from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys

class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--demo', default='', 
                             help='path to image/ image folders/ video. '
                                  'or "webcam"')
        self.parser.add_argument('--load_model', default='',
                                help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                help='resume an experiment. '
                                    'Reloaded the optimizer parameter and '
                                    'set load_model to model_last.pth '
                                  'in the exp dir if load_model is empty.')
        self.parser.add_argument('--gpus', default='0', 
                             help='-1 for CPU, use comma for multiple gpus')
        
        self.parser.add_argument('--input_res', type=int, default=-1, 
                                help='input height and width. -1 for default from '
                                'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1, 
                                help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1, 
                                help='input width. -1 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4, 
                                help='learning rate for batch size 32.')
        self.parser.add_argument('--lr_step', type=str, default='90,120',
                                help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=140,
                                help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                help='batch size')
        self.parser.add_argument('--master_batch_size', type=int, default=-1,
                                help='batch size on the master gpu.')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=5,
                                help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                help='include validation in training and '
                                    'test on test set')
        self.parser.add_argument('--dataset', default='',
                                help='train data set')
        self.parser.add_argument('--metric', default='loss', 
                            help='main metric to save best model')

        # test
        self.parser.add_argument('--flip_test', action='store_true',
                                help='flip data augmentation.')
        self.parser.add_argument('--test_scales', type=str, default='1',
                                help='multi scale test augmentation.')
        self.parser.add_argument('--nms', action='store_true',
                                help='run nms in testing.')
        self.parser.add_argument('--K', type=int, default=100,
                                help='max number of output objects.') 
        self.parser.add_argument('--not_prefetch_test', action='store_true',
                                help='not use parallal data pre-processing.')
        self.parser.add_argument('--fix_res', action='store_true',
                                help='fix testing resolution or keep '
                                    'the original resolution')
        self.parser.add_argument('--keep_res', action='store_true',
                                help='keep the original resolution'
                                    ' during validation.')

        #lane det
        self.parser.add_argument('--delta_v', type=float, default=0.5,
                                help='Discriminative loss delta_v')
        self.parser.add_argument('--delta_d', type=float, default=3.0,
                                help='Discriminative loss delta_d')
        self.parser.add_argument('--lane_emdebing_dim', type=int, default=4,
                                help='lane segmentation embeding dim')
        self.parser.add_argument('--lane_feature_dim',type=int, default=64,
                                help = 'lane segmengtation feature map dim')
        self.parser.add_argument('--lane_loss_scale', type=float, default=1.0,
                                help='lane segmentation loss scale')
        self.parser.add_argument('--lane_loss_var_scale', type=float, default=1.0,
                                help='Discriminative loss var scale')
        self.parser.add_argument('--lane_loss_dist_scale', type=float, default=1.0,
                                help='Discriminative loss dist scale')
        self.parser.add_argument('--lane_loss_reg_scale', type=float, default = 0.01,
                                help='Discriminative loss reg scale')

    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)
        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(len(opt.gpus))] if opt.gpus[0] >=0 else [-1]
        opt.lr_step = [int(i) for i in opt.lr_step.split(',')]
        opt.head_conv = 256
        opt.num_stacks = 1
        opt.pad = 31
        if opt.master_batch_size == -1:
            opt.master_batch_size = opt.batch_size // len(opt.gpus)

        rest_batch_size = (opt.batch_size - opt.master_batch_size)
        opt.chunk_sizes = [opt.master_batch_size]
        for i in range(len(opt.gpus) - 1):
            slave_chunk_size = rest_batch_size // (len(opt.gpus) - 1)
            if i < rest_batch_size % (len(opt.gpus) - 1):
                slave_chunk_size += 1
            opt.chunk_sizes.append(slave_chunk_size)
        print('training chunk_sizes:', opt.chunk_sizes)

        opt.root_dir = opt.dataset
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        opt.lmdb_dir = os.path.join(opt.root_dir, 'data')

        opt.heads = {'binary': 2, 'segmentation': 64}

        return opt

        