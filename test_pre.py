# -- coding:utf-8 --
import os
import numpy as np
import argparse
from utils.metrics import get_MAE, get_MSE, get_MAPE
import torch
from prediction import TransAm
from sr import Generator
from data_process_pre import get_dataloader_pre
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
# load arguments
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='which device to use')
parser.add_argument('--upscale_factor', type=int, default=4, help='upscale factor')
parser.add_argument('--num_epochs', default=300, type=int, help='train epoch number')
# 1500 和 100:根据统计数据后粗细粒度每个格子大概的信号容量制定
parser.add_argument('--scaler_X', type=int, default=1500, help='scaler of coarse-grained flows')
parser.add_argument('--scaler_Y', type=int, default=100, help='scaler of fine-grained flows')
parser.add_argument('--n_residuals', type=int, default=8, help='number of residual units')
parser.add_argument('--base_channels', type=int, default=64, help='number of feature maps')
parser.add_argument('--batch_size', type=int, default=32,
                    help='training batch size')
parser.add_argument('--dataset', type=str, default='P6',
                    help='which dataset to use')
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
parser.add_argument('--nb_flow', type=int, default=2)
parser.add_argument('--map_height', type=int, default=40)
parser.add_argument('--map_width', type=int, default=16)

opt = parser.parse_args()
print(opt)

# test CUDA
cuda = True if torch.cuda.is_available() else False
device = torch.device(opt.device)
"""
测试过程
"""
def test_pre(mode):
    model_path = 'saved_model/separate/{}/cpt/{}-{}-{}_{}_{}_{}'.format(opt.dataset,
                                                                     opt.len_closeness,
                                                                     opt.len_period,
                                                                     opt.len_trend,
                                                                     opt.n_heads,
                                                                     opt.num_layers,
                                                                    opt.skip_dim)
    model = TransAm(in_channel=opt.nb_flow,feature_size=opt.feature_size, hid_dim=opt.hidden_dim, n_heads=opt.n_heads, dim_head=opt.dim_head,
                           skip_dim=opt.skip_dim, num_layers=opt.num_layers,
                           len_clossness=opt.len_closeness, len_period=opt.len_period, len_trend=opt.len_trend,
                           map_heigh=opt.map_height,map_width=opt.map_width,ext_flag=False,external_dim=opt.external_dim, dropout=opt.dropout)
   # 得到网络实例模型
    model.load_state_dict(torch.load('{}/final_model.pt'.format(model_path)))
    model.eval()
    save_path = os.path.join('data/{}'.format(opt.dataset), mode)
    os.makedirs(save_path, exist_ok=True)


    MSE = 100000
    datapath = os.path.join('data', opt.dataset)
    dataloader = get_dataloader_pre(
        datapath, opt.len_closeness, opt.len_period, opt.len_trend, opt.scaler_X,  1,False,
        mode='test',map_H=opt.map_height,map_W=opt.map_width,day_len=24,channel=opt.nb_flow)#24,48
    total_mse, total_mae, total_mape, total_rmse = 0, 0, 0, 0

    # for n, (xc, xp, xt, ext, lable) in enumerate(dataloader):
    for n, (xc, xp, xt,  lable) in enumerate(dataloader):
        los = 0
        l = lable.to(device)
        xc = xc.to(device)
        xp = xp.to(device)
        xt = xt.to(device)
        model = model.to(device)
        # ext = ext.to(device)
        B, Tc,C, H, W = xc.shape
        with torch.no_grad():
            # out = model(xc, xp, xt, ext)
            out = model(xc, xp, xt)
            # 计算和预期输出之间的MSE损失
        pre = out.cpu().detach().numpy() * opt.scaler_X
        l = l.cpu().detach().reshape(B,C, H, W).numpy() * opt.scaler_X
        total_mse += get_MSE(pre, l) * B
        total_mae += get_MAE(pre, l) * B
        total_mape += get_MAPE(pre, l) * B


    mse = total_mse / len(dataloader.dataset)
    mae = total_mae / len(dataloader.dataset)
    mape = total_mape / len(dataloader.dataset)
    rmse = np.sqrt(mse)

    f = open('{}/results.txt'.format(model_path), 'a')
    f.write("{}:\tRMSE={:.6f}\tMAE={:.6f}\tMAPE={:.6f}\n".format(mode,rmse, mae, mape))
    f.close()

    print('Test MSE = {:.6f} ,MAE = {:.6f}, MAPE = {:.6f},RMSE = {:.6f}'.format(mse, mae, mape, rmse))


if __name__ == '__main__':
    # test_pre('train')
    # test_pre('valid')
    test_pre('test')

