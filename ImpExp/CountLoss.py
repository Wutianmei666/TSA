from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.imp_args import _make_imp_args
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import datetime
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

warnings.filterwarnings('ignore')

# 单独训练
class Exp_Count_Imp_Loss(Exp_Basic):
    def __init__(self, args):
        super(Exp_Count_Imp_Loss, self).__init__(args)
        self.args = args
        self.imp_model, self.img_args = self._bulid_imputation_model()
        print("Using {} to imputate data".format(self.imp_args.model))

    def _build_model(self):
        return

    def _bulid_imputation_model(self):
        imp_args, weight_path = _make_imp_args(self.args)
        imp_model = self.model_dict[imp_args.model].Model(imp_args).float()

        assert weight_path != '', '需加载填补模型权重'
        # 装载填补模型权重
        imp_model.load_state_dict(torch.load(weight_path))
        imp_model.to(self.device)
        imp_model.eval()
        return imp_model, imp_args

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

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        # if test:
        #     print('loading model')
        #     self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        
        imp_mse = []
        imp_mae = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            mse_fn = nn.MSELoss()
            mae_fn = nn.L1Loss()
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x_raw = batch_x_raw.float().to(self.device)
                batch_y_raw = batch_y_raw.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                ##  复制一个batch_x用于后续计算填补损失
                batch_x_raw_clone = batch_x_raw.clone().detach().cpu()

                ## 填补
                # random mask
                B, T, N = batch_x_raw.shape
                mask = torch.rand((B, T, N)).to(self.device)
                mask[mask <= self.args.mask_rate] = 0  # masked
                mask[mask > self.args.mask_rate] = 1  # remained
                inp = batch_x_raw.masked_fill(mask == 0, 0)

                # 输入
                batch_x_imp = self.imp_model(inp, batch_x_mark, None, None, mask)

                # 补回去被填充的部分
                batch_x_imp = batch_x_raw*mask + batch_x_imp*(1-mask)

                # 复制一个
                batch_x_imp_clone = batch_x_imp.clone().detach().cpu()
                # 将batch_x_imp属于label_len部分赋值给batch_y
                #batch_y_imp = torch.cat([batch_x_imp[:,-self.args.label_len:,:],batch_y_raw[:,-self.args.pred_len:,:]],dim=1).float().to(self.device)
                
                ## 预测
                # decoder input
                dec_inp = torch.zeros_like(batch_y_raw[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_x_imp[:, -self.args.label_len:, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y_raw = batch_y_raw[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y_raw = batch_y_raw.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = test_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                    batch_y_raw = test_data.inverse_transform(batch_y_raw.squeeze(0)).reshape(shape)
        
                outputs = outputs[:, :, f_dim:]
                batch_y_raw = batch_y_raw[:, :, f_dim:]

                pred = outputs
                true = batch_y_raw

                imp_mae.append(mae_fn(batch_x_raw_clone[mask==0],batch_x_imp_clone[mask==0]).item())
                imp_mse.append(mse_fn(batch_x_raw_clone[mask==0],batch_x_imp_clone[mask==0]).item())

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     raw = batch_x_raw.detach().cpu().numpy()
                #     input = batch_x_imp.detach().cpu().numpy()
                #     if test_data.scale and self.args.inverse:
                #         shape = input.shape
                #         raw = test_data.inverse_transform(raw.squeeze(0)).reshape(shape)
                #         input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                #     gt = np.concatenate((raw[0, :, -1], true[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # calculate imputaion loss
        imp_mae = np.mean(imp_mae)
        imp_mse = np.mean(imp_mse)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1,1)
                y = trues[i].reshape(-1,1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = -999
            

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f = open("result_long_term_forecast_imp_i.txt", 'a')
        f.write(setting + "  \n")
        f.write("Use {} to imputate data \n".format(self.imp_args.model))
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)


        # 写入csv文件中,需要保存的参数如下
        """ 数据集:self.args.dataset  
            填补模型: self.imp_args.model
            下游模型:self.args.model 
            掩码率:str(self.args.mask_rate*100)+'%',
            填补mse:imp_mse ,
            填补mae:imp_mae,
            下游mse:mse ,
            下游mae:mae ,
            "种子": self.args.random_seed,
            "日期":datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'),
            "是否汇入总表":0
        """

        df = pd.read_csv('I.csv')
        result_dict = {
                        "数据集":self.args.dataset,
                        "填补模型": self.imp_args.model,
                        "填补模型d_model": self.imp_args.d_model,
                        "填补模型d_ff":self.imp_args.d_ff,
                        "下游模型": self.args.model, 
                        "掩码率":str(self.args.mask_rate*100)+'%',
                        "填补MSE":imp_mse ,
                        "填补MAE":imp_mae,
                        "下游MSE":mse ,
                        "下游MAE":mae ,
                        "种子": self.args.random_seed,
                        "日期":datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S'),
                        "是否汇入总表":0
                        }
        df = pd.concat([df,pd.DataFrame([result_dict])],ignore_index=True)
        df.to_csv('I.csv',index=False)

        return
