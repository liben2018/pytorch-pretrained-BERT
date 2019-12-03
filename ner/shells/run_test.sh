# Run BERT-NER codes

# Set which GPU to be used
export CUDA_VISIBLE_DEVICES=0 #,1,2,3

export ROOT=/home/ben/transformers
export MAX_LENGTH=512
export SEED=1
export cache_dir=$ROOT/weights/
# batch_size=4 for large model for one GPU, and batch_size=8 for base model.
export BATCH_SIZE=4
export BERT_MODEL_base_cased=bert-base-multilingual-cased # don't need do_lower_case, cased: A and a are different!
export BERT_MODEL_base_uncased=bert-base-multilingual-uncased # need do_lower_case, uncased: A and a are same!
export BERT_MODEL_large_uncased=bert-large-uncased-whole-word-masking # need do_lower_case
export BERT_MODEL_large_cased=bert-large-cased-whole-word-masking # don't need do_lower_case

export data_dir=$ROOT/datasets/jpner_1119_split
export label_file=label_invoice_ner.txt
export SAVE_STEPS=5000

export NUM_EPOCHS=200
export OUTPUT_DIR=$ROOT/ner/results/jpner_test_1126_e200

python3 ner_test.py \
--data_dir $data_dir \
--model_type bert \
--labels $data_dir/$label_file \
--output_dir $OUTPUT_DIR \
--max_seq_length  $MAX_LENGTH \
--num_train_epochs $NUM_EPOCHS \
--max_steps -1 \
--per_gpu_train_batch_size $BATCH_SIZE \
--per_gpu_eval_batch_size $BATCH_SIZE \
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
--cache_dir $cache_dir \
--model_name_or_path $BERT_MODEL_base_cased \
# --do_lower_case \
# --local_rank 1 \
