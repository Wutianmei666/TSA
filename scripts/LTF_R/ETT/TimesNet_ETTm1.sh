export CUDA_VISIBLE_DEVICES=0
seed=2024
model_name=TimesNet
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
d_model=64
d_ff=64
top_k=5
for interpolate in mean nearest linear
do
    for mask_rate in  0.125 0.25 0.375 0.5 0.625 0.75
    do
        python -u mix_run.py \
          --random_seed $seed \
          --task_name long_term_forecast \
          --train_mode 2 \
          --mask_rate $mask_rate \
          --interpolate $interpolate \
          --is_training 1 \
          --model_id ETTm1_${mask_rate}_96_96_R_${interpolate}\
          --root_path ./dataset/ETT-small/ \
          --data_path ETTm1.csv \
          --model $model_name \
          --dataset ETTm1 \
          --data ETTm1 \
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
          --des 'Exp' \
          --itr 1 \
          --top_k $top_k
    done 
done