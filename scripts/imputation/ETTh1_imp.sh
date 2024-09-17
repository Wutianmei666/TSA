export CUDA_VISIBLE_DEVICES=1
model_name=FEDformer
d_model=64
d_ff=32
seq_len=96
label_len=0
pred_len=0
learning_rate=0.001
lradj=type1
#for mask_rate in 0.125 0.25 0.375 0.5 0.625 0.75
#for mask_rate in 0.125
for mask_rate in 0.25 0.375 0.5 0.625 0.75
do
  python -u run.py \
    --task_name imputation \
    --is_training 1 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id ETTh1_mask_${mask_rate} \
    --mask_rate $mask_rate \
    --model $model_name \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --batch_size 16 \
    --d_model $d_model \
    --d_ff $d_ff \
    --des 'Exp' \
    --itr 1 \
    --top_k 5 \
    --lradj $lradj \
    --learning_rate $learning_rate \
    --train_epochs 10
done 

