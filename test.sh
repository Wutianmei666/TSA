export CUDA_VISIBLE_DEVICES=0

model_name=TimesNet
for mask_rate in  0.5
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 2 \
      --mask_rate $mask_rate \
      --interpolate mean \
      --is_training 1 \
      --model_id ETTh1_${mask_rate}_96_96_R_mean \
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
      --d_model 32 \
      --d_ff 32 \
      --des 'Exp' \
      --itr 1 \
      --top_k 5 
done
for mask_rate in 0.125 0.25 0.375 0.5
do
    python -u mix_run.py \
    --task_name long_term_forecast \
    --train_mode 2 \
    --mask_rate $mask_rate \
    --interpolate mean \
    --is_training 1 \
    --model_id ETTh1_${mask_rate}_96_96_R_mean \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model MICN \
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
done
for mask_rate in 0.125 0.25 0.375 0.5
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 2 \
      --mask_rate $mask_rate \
      --interpolate mean \
      --is_training 1 \
      --model_id ETTh1_${mask_rate}_96_96_R_mean \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model Transformer \
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
done