# run BERT-NER codes

export ROOT=/home/ben/transformers
export MAX_LENGTH=512
export BATCH_SIZE=16
export SEED=1
export BERT_MODEL=bert-base-multilingual-cased
export cache_dir=$ROOT/weights/

export data_dir=$ROOT/datasets/jpner_1119_split
export label_file=label_invoice_ner.txt
export SAVE_STEPS=5000

export NUM_EPOCHS=1
export OUTPUT_DIR=$ROOT/ner/results/jpner_test_1121

python3 ner_test.py \
--data_dir $data_dir \
--model_type bert \
--labels $data_dir/$label_file \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--max_steps -1 \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir \
--overwrite_cache \
--fp16 \
--fp16_opt_level 'O1' \
--eval_all_checkpoints \
# --cache_dir $cache_dir \
