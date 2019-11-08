from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import __init_path
from tools.opts import opts
from tools.utils import AverageMeter
from tools.model_tool import load_model, save_model
from tools.data_parallel import DataParallel
from lane_net.dlav import get_pose_net, SegmentationInstance
from loss.loss import DiscriminativeLoss, LaneLoss, ModleWithLoss
from dataset.lane_dataloader import DetLaneDataset
import torch
import os

def run_epoch(phase, epoch, data_loader, model_loss, opt, optimizer, losses_stat):
    avg_loss_stats = {k : AverageMeter() for k in losses_stat}
    if phase == 'train':
        model_loss.train()
    else:
        if len(opt.gups) > 1:
            model_loss = model_loss.module
        model_loss.eval()
        torch.cuda.empty_cache()
    for iter_id, batch in enumerate(data_loader):

        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].to(device=opt.device, non_blocking=True)
        output, loss, loss_stats = model_loss(batch)
        loss = loss.mean()
        if phase == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        message = phase + ' | ' + ' epoch : ' + str(epoch) + ' | iter : '+iter_id+" | "
        for kw in loss_stats.items():
            message += kw[0]+ ' : '+ str(kw[1]) + ' | '
        print(message)

        for k in avg_loss_stats:
            avg_loss_stats[k].update(
                loss_stats[k].mean().item(), batch['iinput'].size(0)
            )
        ret = {k: v.avg for k, v in avg_loss_stats.items()}
        return ret
    


def train(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.opt_str
    opt.device = torch.device('cuda' if opt.gups[0] >= 0 else 'cpu')


    #create model
    model = get_pose_net(heads=opt.heads)
    model = SegmentationInstance(model, opt)
    
    #create loss
    loss = LaneLoss(opt)
    model_loss = ModleWithLoss(model, loss)

    #create optim
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=0.001)

    if len(opt.gpus) > 1:
        model_loss = DataParallel(
            model_loss, device_ids=opt.gpus,
            chunk_sizes=opt.chunk_sizes
        ).to(opt.device)
    else:
        model_loss = model_loss.to(opt.device)
    for state in optimizer.state.values():
        for k, v in state.items():
            state[k] = v.to(device=opt.device, non_blocking=True)
    #create Dataset
    #for train
    train_dataloader = torch.utils.data.DataLoader(
        DetLaneDataset(opt, 'train'),
        batch_size = opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    # val 
    val_dataloader = torch.utils.data.DataLoader(
        DetLaneDataset(opt, 'val'),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )

    #load model
    start_epoch = 0
    if opt.model != '':
        model, optimizer, start_epoch = load_model(
            model, opt.load_model, optimizer, opt.resume, opt.lr, opt.lr_step
        )

    print('train ...')
    best = 1e10
    losses_stat =  ['loss', 'seg_loss', 'binary_loss', 'seg_var_loss', 'seg_dist_loss', 'seg_reg_loss' ]
    for epoch in range(start_epoch +1, opt.num_epochs + 1):

        train_ret = run_epoch('train', epoch, train_dataloader, model_loss, opt, optimizer, losses_stat)
        
        if opt.val_intervals > 0 and epoch % opt.val_intervals == 0:
            save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(epoch)), epoch, model, optimizer)
            with torch.no_grad():
                ret = run_epoch('val', epoch, val_dataloader, model_loss, opt, optimizer, losses_stat)
            if ret[opt.metric] < best:
                save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
        else:
            save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
        