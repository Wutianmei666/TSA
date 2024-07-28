export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet

python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 2 \
  --mask_rate 0.5 \
  --interpolate no \
  --is_training 1 \
  --model_id ECL_0.5_96_96_R_no \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model 256 \
  --d_ff 512 \
  --top_k 5 \
  --des 'Exp' \
  --itr 1
