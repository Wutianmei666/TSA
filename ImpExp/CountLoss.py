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
        assert self.args.imp_method in ['mean','nearest','linear','DL'], '选择的填补方法不合规定,可选的有:interpolate,DL'
        self.args = args
        
        if self.args.imp_method == 'DL':
            self.imputation_model, self.imp_args = self._build_imputation_model()
        print("Using {} to imputate data".format(self.args.imp_method))

    def _build_imputation_model(self):
        imp_args, imp_model_weight_path = _make_imp_args(self.args)
        imp_model = self.model_dict[imp_args.model].Model(imp_args).float()

        assert imp_model_weight_path != '', '需加载填补模型权重'
        # 装载填补模型权重
        imp_model.load_state_dict(torch.load(imp_model_weight_path))
        imp_model.to(self.device)
        imp_model.eval()
        return imp_model, imp_args

    # def _build_imputation_method(self):
    #     assert self.args.imp_method in ['interpolate','DL'], '选择的填补方法不合规定,可选的有:interpolate,DL'
    #     if self.args.imp_method == 'interpolate':
    #         if self.args.interpolate == 'mean' :
    #             self.method_type = 'mean'
    #             self.masked_mean = masked_mean()
    #         else :
    #             self.method_type = self.args.interpolate
    #             self.interpolate = interpolate() 
    #     else:
    #         self.method_type = 'DL'
    #         self.imputation_model, self.imp_args = self._build_imputation_model()
    
    def imputation_method(self,batch_x,batch_x_mark,mask):
        if self.args.imp_method == 'mean' :
            return masked_mean(batch_x,mask)
        elif self.args.imp_method == 'DL' :
            return self.imputation_model(batch_x,batch_x_mark,None,None,mask)
        else:
            assert self.args.imp_method in ['nearest','linear']
            return interpolate(batch_x,self.args.imp_method)
        
    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model
    
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
                position_list[T-zero_start][b,zero_start:T+1,:] = True
        return position_list

    def test(self, setting, test=0):
        torch.manual_seed(self.args.random_seed)
        test_data, test_loader = self._get_data(flag='test')
        max_consecutive_length = math.ceil(self.args.pred_len*self.args.mask_rate)
        preds= []
        trues = []
        mse_results = []
        mae_results = []
        #position_list_all = []
        #mask_list = []
        with torch.no_grad():
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x_raw = batch_x_raw.float().to(self.device).detach()
                batch_x_mark = batch_x_mark.float().to(self.device).detach()
                ## 填补
                # random mask
                B, T, N = batch_x_raw.shape
                mask = torch.rand((B, T, 1)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                mask = mask.expand(B,T,N)

                position_list = self.build_position_list(mask,max_consecutive_length)
                #position_list_all.append(position_list)
                #mask_list.append(mask.detach().cpu())
                inp = batch_x_raw.masked_fill(mask == 0, 0)

                # 输出
                batch_x_imp = self.imputation_method(inp,batch_x_mark,mask)
                batch_x_imp = batch_x_imp.cpu()
                batch_x_raw = batch_x_raw.cpu()
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
        
        folder_path = './count_imp_loss'+'/'+self.args.dataset+'_'+str(self.args.pred_len)+'_'+str(self.args.mask_rate)+'/'+(self.imp_args.model if self.args.imp_method =='DL' else self.args.imp_method) + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'mse.npy', np.array(mse_results))
        np.save(folder_path + 'mae.npy',np.array(mae_results))
        #np.save(folder_path + 'position.npy',np.array(position_list_all))
        #np.save(folder_path + 'mask.npy',np.array(mask_list))

        return
