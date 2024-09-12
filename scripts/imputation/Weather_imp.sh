export CUDA_VISIBLE_DEVICES=1
model_name=MICN
d_model=64
d_ff=64
seq_len=96
label_len=0
pred_len=0
learning_rate=0.001
#for mask_rate in 0.125 0.25 0.375 0.5 0.625 0.75
#for mask_rate in 0.125
for mask_rate in 0.25 0.375 0.5 0.625 0.75
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path weather.csv \
    --model_id weather_mask_${mask_rate} \
    --mask_rate $mask_rate \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --batch_size 16 \
    --d_model $d_model \
    --d_ff $d_ff \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 \
    --learning_rate $learning_rate \
    --lradj type1 \
    --train_epochs 10
done 

