# How to generate datasets for jpner?

# Step1: create datasets with splitting long (>512) sequence
python create_datasets_split.py

# Step2: merge label files
python merge_files.py

# Step3: modifies the datasets, such as, dev.txt or test.txt
# Training start!
