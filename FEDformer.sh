# 插值训练
# bash scripts/LTF_R/ETT/TimesNet_ETTm1.sh
# 单独训练
# bash scripts/LTF_I/ETT/TimesNet_ETTm1.sh
# 联合训练
#bash scripts/LTF_J/ETT/TimesNet_ETTm1.sh

export CUDA_VISIBLE_DEVICES=0

model_name=FEDformer

python -u run.py \
  --task_name imputation \
  --is_training 1 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_mask_0.125 \
  --mask_rate 0.125 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 0 \
  --pred_len 0 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --batch_size 16 \
  --d_model 64 \
  --d_ff 32 \
  --des 'Exp' \
  --itr 1 \
  --top_k 5 \
  --learning_rate 0.001 \
  --train_epochs 10 \
  --lradj type1