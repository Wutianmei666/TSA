export CUDA_VISIBLE_DEVICES=0
model_name=MICN
imp_args_json=ImpModelArgs/ETT/MICN_ETTh1.json
imp_lr=0.001
seq_len=96
label_len=96
pred_len=96
e_layers=2
d_layers=1
learning_rate=0.001
# 固定lambda参数为0 0.5 1
for fix_lambda in 0 0.5 1
do
    for mask_rate in 0.125 0.25 0.375 0.5
    do
        python -u mix_run.py \
            --task_name long_term_forecast \
            --train_mode 1 \
            --_lambda $fix_lambda \
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
            --factor 3 \
            --enc_in 7 \
            --dec_in 7 \
            --c_out 7 \
            --des 'Exp' \
            --itr 1
    done
done

# lambda可学习，使用relu函数使lambda>0
for mask_rate in 0.125 0.25 0.375 0.5
do
    python -u mix_run.py \
      --task_name long_term_forecast \
      --train_mode 1 \
      --_lambda 1 \
      --requires_grad \
      --imp_args_json $imp_args_json \
      --mask_rate $mask_rate \
      --imp_lr $imp_lr \
      --is_training 1 \
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
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des 'Exp' \
      --itr 1 
done