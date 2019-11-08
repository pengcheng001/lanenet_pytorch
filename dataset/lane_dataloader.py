from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import torch
import cv2
import os
from PIL import Image
import io
import torch.utils.data as data
import numpy as np
import glob
import lmdb

from .base_dataset_lmdb import BaseDatsetLmdb


class LaneData(BaseDatsetLmdb):
    def __init__(self, dataset_dir=None, lmdb_path = './', pase = 'train'):
        super(LaneData, self)
        self.root_dir = dataset_dir
        self.pase = pase
        self.lmdb_path = lmdb_path
        self.image_data_db_names = ['gt_image', 'gt_binary_image', 'gt_instance_image']
        self.generate_lmdb()
    
    def _process_dir(self):
        gt_image_dir = os.path.join(self.root_dir, self.pase,  "gt_image")
        gt_binary_image_dir = os.path.join(self.root_dir, self.pase, 'gt_binary_image')
        gt_instance_image_dir = os.path.join(self.root_dir, self.pase, 'gt_instance_image')

        assert os.path.exists(gt_image_dir)
        assert os.path.exists(gt_binary_image_dir)
        assert os.path.exists(gt_instance_image_dir)
        self.datset_list = []

        for _gt_image_path in glob.glob('{:s}/*.png'.format(gt_image_dir)):

            images = []
            _gt_image_name = os.path.split(_gt_image_path)[1]
            _gt_binary_image_path = os.path.join(gt_binary_image_dir, _gt_image_name)
            _gt_instance_image_path = os.path.join(gt_instance_image_dir, _gt_image_name)
            images.append(_gt_image_path)
            images.append(_gt_binary_image_path)
            images.append(_gt_instance_image_path)

            self.datset_list.append(images)

    def get_annot(self, data_list):
        return None
    
    def get_images(self, data_list):
        res = {}
        for i in range(len(self.image_data_db_names)):
            res[self.image_data_db_names[i]] = data_list[i]
        return res

class DetLaneDataset(data.Dataset):

    # def _get_border(self, border, size):
    #     i =1
    #     while size - border // i <= border //i:
    #         i *= 2
    #     return border // i

    mean = np.array([0.399764566851, 0.44865293259, 0.505250931783],
                    dtype=np.float32).reshape(1, 1, 3)
    std  = np.array([0.173585140631, 0.173585140631, 0.228047692886],
                    dtype=np.float32).reshape(1, 1, 3)
    def __init__(self, opt, split):
        self.data_dir = opt.data_dir
        self.lmdb_dir = opt.lmdb_dir
        lane_lmdb = LaneData(dataset_dir=self.data_dir, lmdb_path=self.lmdb_dir, pase=split)
        self.lmdb_path = lane_lmdb.GetLmdbPath()
        self.db_names = ['gt_image', 'gt_binary_image', 'gt_instance_image']
        self.env = lmdb.open(self.lmdb_path, max_dbs=4, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        self.gt_image = self.env.open_db('gt_image'.encode())
        self.gt_instance_image = self.env.open_db('gt_instance_image'.encode())
        self.gt_binary_image = self.env.open_db('gt_binary_image'.encode())
        self.mean = np.array([0.399764566851, 0.44865293259, 0.505250931783],
                    dtype=np.float32).reshape(1, 1, 3)
        self.std  = np.array([0.173585140631, 0.173585140631, 0.228047692886],
                    dtype=np.float32).reshape(1, 1, 3)
        self.opt = opt
        self.default_resolution = (opt.input_w, opt.input_h)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat(db=self.gt_image)['entries']

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        idx = str(index).encode()
        res = {}
        with self.env.begin(write=False) as txn:
            gt_img = txn.get(idx, db = self.gt_image)
            gt_binary_image = txn.get(idx, db = self.gt_binary_image)
            gt_instance_image = txn.get(idx, db = self.gt_instance_image)

        #label_size = (int(self.default_resolution[0] / self.down_ratio), int(self.default_resolution[1] / self.down_ratio))
       # print('======')
        gt_image = Image.open(io.BytesIO(gt_img)).convert("RGB").resize(self.default_resolution, Image.ANTIALIAS)
        gt_binary = Image.open(io.BytesIO(gt_binary_image)).convert('RGB').resize(self.default_resolution, Image.NEAREST)
        gt_instance = Image.open(io.BytesIO(gt_instance_image)).convert('RGB').resize(self.default_resolution, Image.NEAREST)
        gt_inp = np.asarray(gt_image, dtype=np.float32)/255.0
        gt_binary = np.asarray(gt_binary, dtype=np.long)
        gt_instance = np.asarray(gt_instance, dtype=np.long)
        gt_inp = (gt_inp - self.mean) / self.std

        # height, width = gt_inp.shape[0], gt_inp.shape[1]
        # c = np.array([width/2., height / 2.], dtype=np.float32)
        # if self.opt.keep_res:
        #     input_h = (height | self.opt.pad) +1
        #     input_w = (width | self.opt.pad) +1
        #     s = np.array([input_w, input_h], dtype=np.float32)
        # else:
        #     s = max(gt_inp.shape[0], gt_inp.shape[1]) * 1.0
        #     input_h, input_w = self.opt.input_h, self.opt.input_w
        # flipped = False
        # #for train
        # if not self.opt.not_rand_crop:
        #     s = s * np.random.choice(np.arange(0.6,1.4,0.1))
        #     w_border = self._get_border(128, gt_inp.shape[1])
        #     h_border = self._get_border(128, gt_inp.shape[0])
        #     c[0] = np.random.randint(low=w_border, high=gt_inp.shape[1] - w_border)
        #     c[1] = np.random.randint(low=h_border, high=gt_inp.shape[0] - h_border)
        # else:
        #     sf = self.opt.scale
        #     cf = self.opt.shift
        #     c[0] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        #     c[1] += s * np.clip(np.random.randn()*cf, -2*cf, 2*cf)
        #     s = s * np.clip(np.random.randn()*sf +1, 1 - sf, 1 + sf)

        # if np.random.random() < self.opt.flip:
        #     gt_inp = gt_inp[:,::-1,:]
        #     c[0] = width - c[0] -1
        #     gt_binary = gt_binary[:,::-1,:]
        #     gt_instance = gt_instance[:,::-1,:]

        # trans_input = get_affine_transform(
        #     c, s, 0, [input_w, input_h])
        # gt_inp = cv2.warpAffine(gt_inp, trans_input,
        #                         (input_w, input_h),
        #                         flags=cv2.INTER_LINEAR)
        # gt_binary = cv2.warpAffine(gt_binary, trans_input,
        #                             (input_w, input_h),
        #                             flags=cv2.INTER_NEAREST)
        # gt_instance = cv2.warpAffine(gt_instance, trans_input,
        #                             (input_w, input_h),
        #                             flags=cv2.INTER_NEAREST)
        gt_binary = gt_binary/255
        gt_inp = gt_inp.transpose(2,0,1)


        res['input'] = (torch.from_numpy(gt_inp).float())  
        binary = torch.from_numpy(np.asarray(gt_binary)[:,:,0]).long()
        res['binary'] = binary
        res['segmentation'] = torch.from_numpy(np.asarray(gt_instance)[:,:,0]).long()
        return res