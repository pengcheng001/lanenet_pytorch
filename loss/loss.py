from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import cv2 as cv

def unsorted_segment_sum(data, segment_ids, num_segments):
  """
  Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

  :param data: A tensor whose segments are to be summed.
  :param segment_ids: The segment indices tensor.
  :param num_segments: The number of segments.
  :return: A tensor of same data type as the data argument.
  """
  assert all([i in data.shape for i in segment_ids.shape]), "segment_ids.shape should be a prefix of data.shape"

  # segment_ids is a 1-D tensor repeat it to have the same shape as data
  if len(segment_ids.shape) == 1:
      s = torch.prod(torch.tensor(data.shape[1:])).cuda().long()
      segment_ids = segment_ids.repeat_interleave(s).view(segment_ids.shape[0], *data.shape[1:])

  assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

  shape = [num_segments] + list(data.shape[1:])
  tensor = torch.zeros(*shape).cuda()
  tensor = tensor.scatter_add(torch.tensor(0).cuda(), segment_ids.cuda(), data.cuda().float())
  tensor = tensor.type(data.dtype)
  return tensor

class DiscriminativeLoss(torch.nn.Module):
  def __init__(self, delta_v, deltal_d, lane_festure_dim, lane_loss_scale, lane_loss_var_scale, lane_loss_dist_scale, lane_loss_reg_scale ):
      super(DiscriminativeLoss,self).__init__()
      self.delta_v = delta_v
      self.delta_d = deltal_d
      self.feature_dim = lane_festure_dim
      self.param_scale = lane_loss_scale
      self.param_var = lane_loss_var_scale
      self.param_dist = lane_loss_dist_scale
      self.param_reg = lane_loss_reg_scale

  def forward(self, feat, label):
      segmetation_label = label 
      segmentation_output = feat
      b, c, w, h = segmentation_output.shape
      segmetation_label_reshape = segmetation_label.view(b,-1).contiguous()
      segmentation_output_reshape = segmentation_output.view(b, c ,-1).contiguous()
      l_var_res, l_dist_res, l_reg_res, loss_res = 0.0, 0.0, 0.0, 0.0
      for i in range(b):
        seg_feature = segmentation_output_reshape[i]
        seg_lable = segmetation_label_reshape[i]
        label, inds, counts = torch.unique(seg_lable, return_inverse=True, return_counts=True)
        seg_sum = unsorted_segment_sum(seg_feature.t(), inds, len(counts))

        counts = torch.reshape(counts, (-1,1)).float()
        mu = seg_sum / counts

        mu_extern = mu[inds]

        distance = torch.norm(torch.abs(mu_extern - seg_feature.t()), dim=1)
        distance = distance - self.delta_v
        distance = torch.clamp(distance, min = 0.)
        distance = distance**2

        l_var = unsorted_segment_sum(distance, inds, len(counts))
        l_var = l_var / counts.float()
        l_var = torch.mean(l_var)

        instance_num = len(counts)
        mu_interleaved_rep = mu.repeat(instance_num, 1)
        mu_band_rep = mu.repeat(1, instance_num)
        mu_band_rep = torch.reshape(mu_band_rep, (instance_num*instance_num, self.feature_dim))

        mu_diff = mu_band_rep - mu_interleaved_rep
        mu_diff = mu_band_rep - mu_interleaved_rep

        intermediate_tensor = torch.sum(torch.abs(mu_diff), dim=1)
        bool_mask = intermediate_tensor != 0
        mu_diff_bool = mu_diff[bool_mask]

        l_dist = torch.norm(mu_diff_bool, dim=1)
        l_dist = 2 * self.delta_d - l_dist
        l_dist = torch.clamp(l_dist, min=0)
        l_dist = l_dist**2
        l_dist = torch.mean(l_dist)

        l_reg = torch.mean(torch.norm(mu, dim=1))

        l_var_res += self.param_var * l_var
        l_dist_res += self.param_dist * l_dist
        l_reg_res += self.param_reg * l_reg
      loss_res += (l_var_res + l_dist_res + l_reg_res)
      return loss_res/b, l_var_res/b, l_dist_res/b, l_reg_res/b

class emdeding(torch.nn.Module):
  def __init__(self, emdeding_feats_dim = 4, feature_dim = 64):
    super(emdeding, self).__init__()
    self.emdeding_feats_dim = emdeding_feats_dim
    self.pix_bn = torch.nn.BatchNorm2d(num_features = feature_dim)
    self.pix_relu = torch.nn.ReLU()
    self.pix_emdebing = torch.nn.Conv2d(in_channels=feature_dim, out_channels=emdeding_feats_dim, kernel_size=1, bias=False)

  def forward(self, x):
    x = self.pix_bn(x)
    x = self.pix_relu(x)
    x = self.pix_emdebing(x)
    return x
class SegmentationInstance(torch.nn.Module):
    def __init__(self, base_net, opt):
        super(SegmentationInstance, self).__init__()
        self.base_net = base_net
        self.opt = opt
        self.embed = emdeding(opt.lane_emdebing_dim, opt.lane_feature_dim)
    def forward(self, x):
        opt = self.opt
        output = self.base_net(x)
        x = output[-1]
        x['segmentation'] = self.embed(x['segmentation'])
        return [x]


class LaneLoss(torch.nn.Module):
    def __init__(self, opt):
        super(LaneLoss, self).__init__()
        self.opt = opt
        self.discriminativeLoss = DiscriminativeLoss(delta_v=opt.delta_v, deltal_d=opt.delta_d, lane_festure_dim= opt.lane_emdebing_dim, 
                                                    lane_loss_scale = opt.lane_loss_scale, lane_loss_var_scale=opt.lane_loss_var_scale,
                                                    lane_loss_dist_scale=opt.lane_loss_dist_scale, lane_loss_reg_scale=opt.lane_loss_reg_scale)
        weight = torch.FloatTensor([10.,2.5]).cuda()
        self.cross_entropy = torch.nn.CrossEntropyLoss(weight=weight)
        

    
    def forward(self, outputs, batch):
        opt = self.opt
        binary_loss, seg_loss = 0.0, 0.0
        seg_loss_var, seg_loss_dist, seg_loss_reg = 0.0, 0.0, 0.0
        

        for s in range(opt.num_stacks):
            output = outputs[s]
            binay_label = batch['binary']
            binary_output = output['binary']
            n,h,w = binay_label.shape


            binary_loss_ = self.cross_entropy(binary_output, binay_label) / opt.num_stacks
            binary_loss += binary_loss_

            #for segmentatioin

            segmetation_label = batch['segmentation']
            segmentation_output = output['segmentation']
            loss_toal, loss_var, loss_dist, loss_reg = self.discriminativeLoss(segmentation_output, segmetation_label)

            seg_loss_var += loss_var/ opt.num_stacks

            seg_loss_dist += loss_dist/ opt.num_stacks

            seg_loss_reg += loss_reg/ opt.num_stacks

            seg_loss += loss_toal / opt.num_stacks
        loss = seg_loss + 6*binary_loss

        loss_stats = {'loss':loss, 'seg_loss': seg_loss, 'binary_loss': binary_loss, 'seg_var_loss': seg_loss_var, 'seg_dist_loss': seg_loss_dist, 'seg_reg_loss': seg_loss_reg }

        return  loss, loss_stats

class ModleWithLoss(torch.nn.Module):
  def __init__(self, model, loss):
    super(ModleWithLoss, self).__init__()
    self.model = model
    self.loss = loss
  
  def forward(self, batch):
    outputs = self.model(batch['input'])
    loss, loss_stats = self.loss(outputs, batch)
    return outputs[-1], loss, loss_stats
 