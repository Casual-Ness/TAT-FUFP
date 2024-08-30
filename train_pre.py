# -- coding:utf-8 --
#train prediction and super-resolution separately

import argparse
import os
import numpy as np
from math import log10,sqrt
from tqdm import tqdm
import torch.optim as optim
import torch.utils.data
from torch import nn
from datetime import datetime
import pytorch_ssim
from data_process_pre import get_dataloader_pre
from prediction import TransAm
from sr import Generator,Discriminator

parser = argparse.ArgumentParser(description='Train Super Resolution Models')
parser.add_argument('--device', type=str, default='cuda:0', help='which device to use')
parser.add_argument('--upscale_factor', type=int, default=2, help='upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
parser.add_argument('--dataset', type=str, default='P6', help='which dataset to use')
# 1500 和 100:根据统计数据后粗细粒度每个格子大概的信号容量制定
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--batch_size', type=int, default=32, help='training batch size')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--nb_flow', type=int, default=2)
parser.add_argument('--map_height', type=int, default=40)
parser.add_argument('--map_width', type=int, default=16)
#prediction
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=3)
parser.add_argument('--len_trend', type=int, default=0)
parser.add_argument('--external_dim', type=int, default=7)
parser.add_argument('--n_heads', type=int, default=2,
                    help='number of heads of selfattention')
parser.add_argument('--dim_head', type=int, default=8,
                    help='dim of heads of selfattention')
parser.add_argument('--dropout', type=float, default=0,
                    help='encoder dropout')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of encoder layers')
parser.add_argument('--feature_size', type=int, default=64)
parser.add_argument('--hidden_dim', type=int, default=128,
                    help='dim of FC layer')
parser.add_argument('--skip_dim', type=int, default=256,
                    help='dim of skip conv')

# training skills
parser.add_argument('--lr', type=float, default=1e-3, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of second order momentum of gradient')
parser.add_argument('--harved_epoch', type=int, default=20, help='halved at every x interval')
parser.add_argument('--seed', type=int, default=2024, help='random seed')


def get_RMSE(pred, real):
    mse = np.mean(np.power(real - pred, 2))
    return sqrt(mse)

def train_pre(lr,epoch_num):
    # 设置神经网络参数随机初始化种子，使每次训练初始参数可控
    torch.cuda.manual_seed(opt.seed)
    device = torch.device(opt.device)
    rmses = [np.inf]
    # save_path = 'saved_model/separate/{}/cpt_NT/{}-{}-{}_{}_{}_{}'.format(opt.dataset,
    save_path = 'saved_model/separate/{}/cpt/{}-{}-{}_{}_{}_{}'.format(opt.dataset,
                                                               opt.len_closeness,
                                                               opt.len_period,
                                                               opt.len_trend,
                                                               opt.n_heads,
                                                               opt.num_layers,
                                                                opt.skip_dim)
    os.makedirs(save_path, exist_ok=True)
    datapath = os.path.join('data', opt.dataset)

    train_dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  opt.batch_size,False,
        mode='train',map_H=opt.map_height,map_W=opt.map_width,day_len=24,channel=opt.nb_flow)  # opt.batch_size=16
    valid_dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  4, False,
        mode='valid',map_H=opt.map_height,map_W=opt.map_width,day_len=24,channel=opt.nb_flow)#48,24

    pre = TransAm(in_channel=opt.nb_flow,feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           map_heigh=opt.map_height,map_width=opt.map_width,ext_flag=False,external_dim=opt.external_dim, dropout=opt.dropout)
    print('# prediction parameters:', sum(param.numel() for param in pre.parameters()))


    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        print("CUDA可用，正在用GPU运行程序")
        pre.to(device)
        criterion.to(device)

    optimizer = optim.Adam(pre.parameters(), lr=lr, betas=(opt.b1, opt.b2))
    iter = 0
    valid_rmse = torch.zeros((epoch_num,))
    for epoch in range(epoch_num):
        pre.train()  # model.train()：启用Batch_Normalization和Dropout
        train_loss = 0
        ep_time = datetime.now()
        """生成样本数据"""
        # for z, (xc,xp,xt,ext,next) in enumerate(train_dataloader):
        for z, (xc, xp, xt, next) in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss = 0
            B,Tc,_,H,W = xc.shape
            if torch.cuda.is_available():
                xc = xc.to(device)
                xp = xp.to(device)
                xt = xt.to(device)
                # ext = ext.to(device)
                next = next.to(device)
            # pred = pre(xc,xp,xt,ext)
            pred = pre(xc, xp, xt)
            loss = criterion(pred, next.reshape(B,opt.nb_flow,H,W))
            loss.requires_grad_(True)
            # 更新网络参数
            loss.backward()
            optimizer.step()
            print("[Epoch %d/%d] [Batch %d/%d] [Batch Loss: %f]" % (epoch,
                                                                    epoch_num,
                                                                    z,
                                                                    len(train_dataloader),
                                                                    np.sqrt(loss.item())
                                                                    ))

            # counting training mse
            train_loss += loss.item()

            iter += 1
        # validation phase
        with torch.no_grad():
            pre.eval()
            valid_time = datetime.now()
            total_mse = 0
            # for n, (xc, xp, xt, ext, next) in enumerate(valid_dataloader):
            for n, (xc, xp, xt, next) in enumerate(valid_dataloader):
                los = 0
                if torch.cuda.is_available():
                    xc = xc.to(device)
                    xp = xp.to(device)
                    xt = xt.to(device)
                    # ext = ext.to(device)
                Bv, Tv, _, H, W = xc.shape
                # pred = pre(xc, xp, xt, ext).cpu()
                pred = pre(xc, xp, xt).cpu()
                # 计算和预期输出之间的MSE损失
                los = criterion(pred, next.reshape(Bv, opt.nb_flow, H, W))
                total_mse += los * Bv
            rmse = np.sqrt(total_mse / len(valid_dataloader.dataset)) * opt.scaler_X
            valid_rmse[epoch] = rmse
            if rmse < np.min(rmses):
                print("iter\t{}\tRMSE\t{:.6f}\ttime\t{}".format(iter, rmse, datetime.now() - valid_time))
                torch.save(pre.state_dict(),
                           '{}/final_model.pt'.format(save_path))
            rmses.append(rmse)

        # half the learning rate
        if epoch % opt.harved_epoch == 0 and epoch != 0:
            lr /= 2
            optimizer = optim.Adam(pre.parameters(), lr=lr)

        print('=================time cost: {}==================='.format(
            datetime.now() - ep_time))
    np.save(os.path.join(save_path, 'valid_rmse.npy'), valid_rmse.numpy())
    return




if __name__ == '__main__':
    opt = parser.parse_args()
    print(opt)
    UPSCALE_FACTOR = opt.upscale_factor
    NUM_EPOCHS = opt.num_epochs
    train_pre(opt.lr,opt.num_epochs)