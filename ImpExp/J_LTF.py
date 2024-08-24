from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping_J, adjust_learning_rate_J, visual
from utils.imp_args import _make_imp_args
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import copy
import warnings
import numpy as np
import torch.nn.functional as F
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

class joint_loss(nn.Module):
    def __init__(self,_lambda,imp_ls_fn,ds_ls_fn):
        super(joint_loss,self).__init__()
        self._lambda = _lambda
        self.imp_ls_fn = imp_ls_fn
        self.ds_ls_fn = ds_ls_fn
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self,x_imp,y_imp,x_ds,y_ds):
        imp_loss = self.imp_ls_fn(x_imp, y_imp)
        ds_loss = self.ds_ls_fn(x_ds, y_ds)
        if self._lambda.requires_grad == True :
            # 将lambda固定在0~1间
            #self._lambda.data.copy_(self.sigmoid(self._lambda))
            #total_loss = self.sigmoid(self._lambda)*imp_loss + (1-self.sigmoid(self._lambda))*ds_loss
            total_loss = self._lambda*imp_loss + (1-self._lambda)*ds_loss
        else :
            total_loss = self._lambda*imp_loss + ds_loss
        return total_loss, imp_loss, ds_loss

# 联合训练
class Exp_Long_Term_Forecast_Imp_J(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Imp_J, self).__init__(args)
        self.args = args
        self.imp_model, self.imp_model_name = self._bulid_imputation_model()
        print("Using {} to imputate data".format(self.imp_model_name))

        self._lambda = self._build_lambda()
        self.activate_fn = nn.ReLU()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _bulid_imputation_model(self):
        imp_args, weight_path = _make_imp_args(self.args)
        imp_model = self.model_dict[imp_args.model].Model(imp_args)
        imp_model.to(self.device)
        return imp_model, imp_args.model

    def _build_lambda(self):
        _lambda = torch.FloatTensor([self.args._lambda]).to(self.device)
        _lambda = nn.Parameter(_lambda, requires_grad=self.args.requires_grad)
        return _lambda
    
    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.requires_grad == False:
            model_optim = optim.Adam([{'params':self.model.parameters()},
                                    {'params':self.imp_model.parameters(),'lr':self.args.imp_lr}],
                                    lr=self.args.learning_rate)
        else:
            model_optim = optim.Adam([
                                        {'params': self.model.parameters()},
                                        {'params': self.imp_model.parameters(), 'lr': self.args.imp_lr},
                                        {'params': self._lambda, 'lr': 0.0001, 'fixed_lr': False}  
                                    ], lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        imp_loss_fn = nn.MSELoss()
        ds_ls_fn = nn.MSELoss()
        criterion = joint_loss(self._lambda,imp_loss_fn, ds_ls_fn)
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        imp_loss_total = []
        ds_loss_total = []
        self.imp_model.eval()
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x_raw = batch_x_raw.float().to(self.device)
                batch_y_raw = batch_y_raw.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                # 复制一个用于后续损失计算
                x_imp = batch_x_imp.clone()             
                
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
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_raw = batch_y_raw[:, -self.args.pred_len:, f_dim:]

                x_imp = x_imp[mask==0].detach().cpu()
                batch_x_raw = batch_x_raw[mask==0].detach().cpu()
                pred = outputs.detach().cpu()
                true = batch_y_raw.detach().cpu()

                loss, imp_loss, ds_loss = criterion(x_imp,batch_x_raw, pred, true)

                total_loss.append(loss.detach().cpu().item())
                imp_loss_total.append(imp_loss.detach().cpu().item())
                ds_loss_total.append(ds_loss.detach().cpu().item())
        total_loss = np.average(total_loss)
        imp_loss_total = np.average(imp_loss_total)
        ds_loss_total = np.average(ds_loss_total)
        self.model.train()
        self.imp_model.train()
        return total_loss, imp_loss_total, ds_loss_total

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping_J(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.imp_model.train()
            epoch_time = time.time()
            
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x_raw = batch_x_raw.float().to(self.device)
                batch_y_raw = batch_y_raw.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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
                # 复制一个用于后续损失计算
                x_imp = batch_x_imp.clone()

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

                        f_dim = -1 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y_raw = batch_y_raw[:, -self.args.pred_len:, f_dim:]
                        # 将lambda限制在大于0 
                        # if self.args.requires_grad :
                        #     self._lambda.data = self.activate_fn(self._lambda)
                        loss, imp_loss, ds_loss = criterion(x_imp[mask==0],batch_x_raw[mask==0],outputs, batch_y_raw)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_raw = batch_y_raw[:, -self.args.pred_len:, f_dim:]
                    # 将lambda限制在大于0 
                    # if self.args.requires_grad :
                    #     self._lambda.data = self.activate_fn(self._lambda)
                    loss, imp_loss, ds_loss = criterion(x_imp[mask==0],batch_x_raw[mask==0],outputs, batch_y_raw)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | lambda: {2:.7f} | total_loss: {3:.7f} | imp_loss: {4:.7f} | ds_loss: {5:.7f}".format(i + 1, epoch + 1, self._lambda.item(), 
                                                                                                                                            loss.item(),imp_loss.item(),ds_loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_loss_imp, vali_loss_ds = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_loss_imp, test_loss_ds = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            print("Epoch: {0}, Steps: {1} | Vali Imp Loss: {2:.7f} Vali DS Loss: {3:.7f} | Test Imp Loss: {4:.7f} Test DS Loss: {5:.7f}".format(epoch+1,train_steps,
                                                                                                                                                vali_loss_imp,vali_loss_ds,
                                                                                                                                                test_loss_imp,test_loss_ds))
            early_stopping(vali_loss_ds, self.imp_model,self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate_J(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.imp_model.load_state_dict(torch.load(best_model_path)['imp_model'])
        self.model.load_state_dict(torch.load(best_model_path)['ds_model'])

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.imp_model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))['imp_model'])
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))['ds_model'])


        imp_mse = []
        imp_mae = []
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self.imp_model.eval()
        self.model.eval()
        with torch.no_grad():
            mse_fn = nn.MSELoss()
            mae_fn = nn.L1Loss()
            for i, (batch_x_raw, batch_y_raw, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x_raw = batch_x_raw.float().to(self.device)
                batch_y_raw = batch_y_raw.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

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

                # 复制一个用于后续损失计算
                x_imp = batch_x_imp.clone()

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

                batch_x_raw = batch_x_raw.detach().cpu()
                x_imp = x_imp.detach().cpu()

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

                imp_mae.append(mae_fn(batch_x_raw[mask==0],x_imp[mask==0]).item())
                imp_mse.append(mse_fn(batch_x_raw[mask==0],x_imp[mask==0]).item())

                preds.append(pred)
                trues.append(true)
                # if i % 20 == 0:
                #     raw = batch_x_raw.numpy()
                #     input = x_imp.numpy()
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
        
        _lambda = self.args._lambda
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('total_mse:{}, total_mae:{}, imp_mse:{}, imp_mae:{}, ds_mse:{}, ds_mae:{}'.format(_lambda*imp_mse+mse,
                                                                                            _lambda*imp_mae+mae,
                                                                                            imp_mse,
                                                                                            imp_mae,
                                                                                            mse, mae))
        f = open("result_long_term_forecast_imp_j.txt", 'a')
        f.write(setting + "  \n")

        f.write('(imp_model:{}, initial lr:{}, initial imp lr: {} lambda:{}({}), lradj:{})\n'.format(self.imp_model_name,
                                                                                                    self.args.learning_rate, 
                                                                                                    self.args.imp_lr,
                                                                                                    self._lambda.item(), 
                                                                                                    'Fix' if self.args.requires_grad == False else 'Trainable',
                                                                                                    self.args.lradj))

        f.write('ds_mse:{}, ds_mae:{}, imp_mse:{}, imp_mae:{}, total_mse:{}, total_mae:{}'.format(mse, 
                                                                                                  mae,
                                                                                                  imp_mse,
                                                                                                  imp_mae,
                                                                                                  _lambda*imp_mse+mse,
                                                                                                  _lambda*imp_mae+mae,
                                                                                                    ))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'total_metrics.npy', np.array([ _lambda*imp_mae+mae,
        #                                                       _lambda*imp_mse+mse, 
        #                                                       ]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return
