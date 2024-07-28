from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import copy
import warnings
import numpy as np
from utils.dtw_metric import dtw,accelerated_dtw
from utils.augmentation import run_augmentation,run_augmentation_single

warnings.filterwarnings('ignore')

# 单独训练
class Exp_Long_Term_Forecast_Imp_I(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast_Imp_I, self).__init__(args)
        self.args = args
        self.imp_model = self._bulid_imputation_model()

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _bulid_imputation_model(self):
        imp_args = copy.deepcopy(self.args)
        imp_args.task_name = 'imputation'
        imp_args.label_len = 0
        imp_args.pred_len = 0
        # 注意修改
        imp_args.d_model = 64
        imp_args.d_ff = 64
        imp_args.top_k = 3
        imp_model = self.model_dict[self.args.model].Model(imp_args)

        # 装载填补模型权重
        imp_model.load_state_dict(torch.load(self.args.imp_model_pt))
        imp_model.to("cuda")
        imp_model.eval()
        return imp_model

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
        
        total_loss = []
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
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y_raw = batch_y_raw[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y_raw.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            self.imp_model.eval()
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
                        loss = criterion(outputs, batch_y_raw)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x_imp, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y_raw = batch_y_raw[:, -self.args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y_raw)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
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
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
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

                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:
                    raw = batch_x_raw.detach().cpu().numpy()
                    input = batch_x_imp.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        raw = test_data.inverse_transform(raw.squeeze(0)).reshape(shape)
                        input = test_data.inverse_transform(input.squeeze(0)).reshape(shape)
                    gt = np.concatenate((raw[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

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
        f.write('mse:{}, mae:{}, dtw:{}'.format(mse, mae, dtw))
        f.write('\n')
        f.write('\n')
        f.close()
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        return
