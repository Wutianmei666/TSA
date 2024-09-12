export CUDA_VISIBLE_DEVICES=1
model_name=MICN
# 更换填补模型时 需修改imp_args_json及对应的json文件
imp_args_json=ImpModelArgs/ECL/MICN.json
seq_len=96
label_len=96
pred_len=96
e_layers=2
d_layers=1
learning_rate=0.0001
d_model=256
d_ff=512
top_k=5
for mask_rate in 0.125 0.25 0.375 0.5 0.625 0.75
do
python -u mix_run.py \
  --task_name long_term_forecast \
  --train_mode 0 \
  --mask_rate $mask_rate \
  --imp_args_json $imp_args_json \
  --is_training 1 \
  --learning_rate $learning_rate \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$mask_rate'_'96_96_I \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --des 'Exp' \
  --itr 1
done