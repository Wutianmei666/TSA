export CUDA_VISIBLE_DEVICES=0
seq_len=96
# 以下参数仅仅为了程序能运行，无需修改
model_name=TimesNet
label_len=48
pred_len=96
e_layers=2
d_layers=1
learning_rate=0.0001
d_model=32
d_ff=32
top_k=5
# 以上参数仅仅为了程序能运行，无需修改
for imp_method in mean nearest linear
do 
  for mask_rate in 0.25 0.5 0.75
  do
  python -u mix_run.py \
    --task_name count_imp_loss \
    --train_mode -1 \
    --mask_rate $mask_rate \
    --imp_method $imp_method \
    --is_training 0 \
    --root_path ./dataset/ETT-small/ \
    --data_path ETTh1.csv \
    --model_id '' \
    --model $model_name \
    --dataset ETTh1 \
    --data ETTh1 \
    --features M \
    --seq_len $seq_len \
    --label_len $label_len \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor 3 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model $d_model \
    --d_ff $d_ff \
    --top_k $top_k \
    --des 'Exp' \
    --itr 1
    done
done