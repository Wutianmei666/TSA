export CUDA_VISIBLE_DEVICES=0
model_name=TimesNet
# 更换填补模型时 需修改imp_args_json及对应的json文件
imp_args_json=ImpModelArgs/Weather/TimesNet.json
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
learning_rate=0.0001
d_model=32
d_ff=32
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
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$mask_rate'_'96_96_I \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len $label_len \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --d_model $d_model \
  --d_ff $d_ff \
  --top_k $top_k \
  --des 'Exp' \
  --itr 1
done