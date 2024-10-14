from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.imp_args import _make_imp_args
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import MSE,MAE
import torch
import datetime
import math
import torch.nn as nn
from torch import optim
import pandas as pd
import os
import time
import copy
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single
from utils.mask_padding import *

warnings.filterwarnings('ignore')

# 单独训练
class Exp_Count_Imp_Loss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Count_Imp_Loss, self).__init__(args)
        self.args = args
        self._build_imputation_method()
        print("Using {} to imputate data".format(self.imp_args.model))

    def _build_imputation_model(self):
        imp_args, weight_path = _make_imp_args(self.args)
        imp_model = self.model_dict[imp_args.model].Model(imp_args).float()

        assert weight_path != '', '需加载填补模型权重'
        # 装载填补模型权重
        imp_model.load_state_dict(torch.load(weight_path))
        imp_model.to(self.device)
        imp_model.eval()
        return imp_model, imp_args

    def _build_imputation_method(self):
        assert self.args.imp_method in ['interpolate','DL'], '选择的填补方法不合规定,可选的有:interpolate,DL'
        if self.args.imp_method == 'interpolate':
            if self.args.interpolate == 'mean' :
                self.method_type = 'mean'
                self.masked_mean = masked_mean()
            else :
                self.method_type = self.args.interpolate
                self.interpolate = interpolate() 
        else:
            self.method_type = 'DL'
            self.imputation_model, self.imp_args = self._build_imputation_model()
    
    def imputation_method(self,batch_x,batch_x_mark,mask,device):
        if self.method_type == 'mean' :
            return masked_mean(batch_x,mask)
        elif self.method_type == 'DL' :
            return self.imputation_model(batch_x,batch_x_mark,None,None,mask)
        else:
            assert self.method_type in ['nearest','linear']
            return self.interpolate(batch_x,device,self.method_type)

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        return 

    def train(self, setting):
        return 

    def build_position_list(self,mask,max_consecutive_length):
        """
        注释以后写
        """
        B,T,N = mask.shape
        position_list = [torch.zeros((B,T,N),dtype=bool) for i in range(max_consecutive_length+1)]
        for b in range(B):
            zero_start = None
            for t in range(T):
                if mask[b,t,0] == 0:
                    if zero_start == None :
                        zero_start = t
                else :
                    if zero_start is not None :
                        position_list[t-zero_start][b,zero_start:t,:] = True
                        zero_start = None
            if zero_start is not None:
                print(zero_start)
                position_list[T-zero_start][b,zero_start:T+1,:] = True
        return position_list

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        max_consecutive_length = math.ceil(T*self.args.mask_rate)
        preds= []
        trues = []
        mse_results = []
        mae_results = []
        with torch.no_grad():
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x_raw = batch_x_raw.float().to(self.device).detach()

                ## 填补
                # random mask
                B, T, N = batch_x_raw.shape
                mask = torch.rand((B, T, 1)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask.expand(B,T,N)

                position_list = self.build_position_list(mask,max_consecutive_length)
                inp = batch_x_raw.masked_fill(mask == 0, 0)

                # 输出
                batch_x_imp = self.imputation_method(inp,batch_x_mark,mask,self.device)
                batch_x_imp = batch_x_imp.cpu()
                
                for consecutive_length in range(1,max_consecutive_length+1):
                    preds.append(np.array(batch_x_imp[position_list[consecutive_length]]))
                    trues.append(np.array(batch_x_raw[position_list[consecutive_length]]))
        
        # 计算每个间隔的填补损失
        for i in range(max_consecutive_length):
            if preds[i].size != 0:
                mse_results.append(MSE(preds[i],trues[i]))
                mae_results.append(MAE(preds[i],trues[i]))
            else :
                mse_results.append(-1)
                mae_results.append(-1)

        # result save
        folder_path = './imp_loss_count'+'/'+self.args.dataset+'_'+self.args.pred_len+'_'+self.args.mask_rate+'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        

        type_path = self.imp_args.model if self.method_type =='DL' else self.method_type + '/'
        np.save(folder_path + type_path + 'mse.npy', np.array(mse_results))
        np.save(folder_path + type_path + 'mae.npy',np.array(mae_results))

        return
