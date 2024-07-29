export CUDA_VISIBLE_DEVICES=0

model_name=Transformer
# 不用插值填补
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.125 \
  --interpolate no \
  --is_training 1 \
  --model_id ETTh1_0.125_96_96_R_no \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.25 \
  --interpolate no \
  --is_training 1 \
  --model_id ETTh1_0.25_96_96_R_no \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \


python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.375 \
  --interpolate no \
  --is_training 1 \
  --model_id ETTh1_0.375_96_96_R_no \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.5 \
  --interpolate no \
  --is_training 1 \
  --model_id ETTh1_0.5_96_96_R_no \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 
  
# 掩码后使用临近插值法填补，再下游
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.125 \
  --interpolate nearest \
  --is_training 1 \
  --model_id ETTh1_0.125_96_96_R_nearest \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.25 \
  --interpolate nearest \
  --is_training 1 \
  --model_id ETTh1_0.25_96_96_R_nearest \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \


python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.375 \
  --interpolate nearest \
  --is_training 1 \
  --model_id ETTh1_0.375_96_96_R_nearest \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.5 \
  --interpolate nearest \
  --is_training 1 \
  --model_id ETTh1_0.5_96_96_R_nearest \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

# 掩码后使用线性插值法填补，再下游
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.125 \
  --interpolate linear \
  --is_training 1 \
  --model_id ETTh1_0.125_96_96_R_linear \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.25 \
  --interpolate linear \
  --is_training 1 \
  --model_id ETTh1_0.25_96_96_R_linear \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 \


python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.375 \
  --interpolate linear \
  --is_training 1 \
  --model_id ETTh1_0.375_96_96_R_linear \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.5 \
  --interpolate linear \
  --is_training 1 \
  --model_id ETTh1_0.5_96_96_R_linear \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 