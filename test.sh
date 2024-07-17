export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 1 \
  --mask_rate 0.125 \
  --_lambda 0.5 \
  --requires_grad True \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_0.125_96_96_J_L \
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
  --d_model 32 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 