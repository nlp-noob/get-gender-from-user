python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name sst2 \
  --do_train True \
  --do_eval True \
  --max_seq_length 128 \
  --per_device_train_batch_size 32 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --no_cuda True \
  --logging_strategy steps \
  --logging_first_step True \
  --logging_steps 10 \
  --evaluation_strategy steps \
  --eval_steps 10 \
  --overwrite_output_dir True \
  --output_dir ./tmp/sst2/ \
