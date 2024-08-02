export CUDA_VISIBLE_DEVICES=0
model_name=Transformer
seq_len=96
label_len=48
pred_len=96
e_layers=2
d_layers=1
d_model=32
d_ff=32
for interpolate in no mean nearest linear
do
    for mask_rate in  0.125 0.25 0.375 0.5
    do
        python -u mix_run.py \
          --task_name long_term_forecast \
          --train_mode 2 \
          --mask_rate $mask_rate \
          --interpolate $interpolate \
          --is_training 1 \
          --model_id ETTh1_${mask_rate}_96_96_R_${interpolate}\
          --root_path ./dataset/ETT-small/ \
          --data_path ETTh1.csv \
          --model $model_name \
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
          --des 'Exp' \
          --itr 1 
    done 
done
