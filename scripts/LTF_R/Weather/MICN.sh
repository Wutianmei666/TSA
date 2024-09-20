export CUDA_VISIBLE_DEVICES=0
model_name=MICN     
seq_len=96
label_len=96
pred_len=96
e_layers=2
d_layers=1
d_model=32
d_ff=32
top_k=5
for interpolate in no mean nearest linear
do
    for mask_rate in  0.125 0.25 0.375 0.5 0.625 0.75
    do
        python -u mix_run.py \
          --task_name long_term_forecast \
          --train_mode 2 \
          --mask_rate $mask_rate \
          --interpolate $interpolate \
          --is_training 1 \
          --model_id weather_${mask_rate}_96_96_R_${interpolate}\
          --root_path ./dataset/weather/ \
          --data_path weather.csv \
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
          --des 'Exp' \
          --itr 1 \
          --top_k $top_k
    done 
done