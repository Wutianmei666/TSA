export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
imp_args_json=ImpModelArgs/ETT/TimesNet_ETTh1.json
imp_lr=0.001
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
d_model=32
d_ff=32
learning_rate=0.0001
top_k=5


for mask_rate in 0.125 0.25 0.375 0.5 0.625 0.75
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 1 \
      --_lambda 0.5 \
      --requires_grad \
      --imp_args_json $imp_args_json \
      --mask_rate $mask_rate \
      --imp_lr $imp_lr \
      --is_training 1 \
      --learning_rate $learning_rate \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${mask_rate}_96_96_J \
      --model $model_name \
      --data ETTh1 \
      --features M \
      --seq_len $seq_len \
      --label_len $label_len \
      --pred_len $pred_len \
      --e_layers $e_layers \
      --d_layers $d_layers \
      --d_model $d_model \
      --d_ff $d_ff \
      --top_k $top_k \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 
done
