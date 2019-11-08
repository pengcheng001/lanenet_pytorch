import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
this_dir = osp.dirname(__file__)
add_path(this_dir)

dataset_path = osp.join(this_dir, 'dataset')
add_path(dataset_path)

lane_net_path = osp.join(this_dir, 'lane_net')
add_path(lane_net_path)

loss_path = osp.join(this_dir, 'loss')
add_path(loss_path)

tools_path = osp.join(this_dir, 'tools')
add_path(tools_path)