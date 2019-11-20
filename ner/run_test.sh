# run BERT-NER codes

export MAX_LENGTH=512
export BERT_MODEL=bert-base-multilingual-cased
export OUTPUT_DIR=jpner_split_test
export BATCH_SIZE=16
export NUM_EPOCHS=400
export SAVE_STEPS=5000
export SEED=1
export data_dir=/home/ben/transformers/datasets/jpner_1119_split
# export data_dir=/home/ben/transformers/datasets/jpner_1106
python3 ner_test.py \
--data_dir $data_dir \
--model_type bert \
--labels $data_dir/label_invoice_ner.txt \
--model_name_or_path $BERT_MODEL \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--per_gpu_train_batch_size $BATCH_SIZE \
--save_steps $SAVE_STEPS \
--seed $SEED \
--do_train \
--do_eval \
--do_predict \
--overwrite_output_dir \
--overwrite_cache \
--fp16 \
