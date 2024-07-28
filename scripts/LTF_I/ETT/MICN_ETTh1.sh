export CUDA_VISIBLE_DEVICES=0

model_name=MICN

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --mask_rate 0.125 \
  --learning_rate 0.001 \
  --imp_model_pt checkpoints/imputation_ETTh1_mask_0.125_MICN_ETTh1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_0.125_96_96_I \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
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
  --train_mode 0 \
  --mask_rate 0.25 \
  --learning_rate 0.001 \
  --imp_model_pt checkpoints/imputation_ETTh1_mask_0.25_MICN_ETTh1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_0.25_96_96_I \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
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
  --train_mode 0 \
  --mask_rate 0.375 \
  --learning_rate 0.001 \
  --imp_model_pt checkpoints/imputation_ETTh1_mask_0.375_MICN_ETTh1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_0.375_96_96_I \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
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
  --train_mode 0 \
  --mask_rate 0.5 \
  --learning_rate 0.001 \
  --imp_model_pt checkpoints/imputation_ETTh1_mask_0.5_MICN_ETTh1_ftM_sl96_ll0_pl0_dm64_nh8_el2_dl1_df64_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0/checkpoint.pth \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_0.5_96_96_I \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --itr 1 