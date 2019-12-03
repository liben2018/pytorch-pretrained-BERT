import os
import sys


def make_new_path(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def merge_file(input1, input2, output_path):
    with open(input1, 'r') as f1, open(input2, 'r') as f2:
        input1_list = f1.readlines()
        input2_list = f2.readlines()
        input_list = input1_list + input2_list
        output_list = list(set(input_list))

    with open(output_path, 'w') as f:
        for label in output_list:
            print('label:', label)
            if label in ['\n', '\t']:
                pass
            else:
                f.write("%s" % label)

    return


def main():
    PATH_DIR = os.getcwd()
    print(PATH_DIR)
    #
    output_dataset_dir = 'jpner_1128_split'
    dataset_dir_path = os.path.join(PATH_DIR, output_dataset_dir)
    make_new_path(dataset_dir_path)
    #
    input_file_1 = "label_invoice_ner_1.txt"
    input_file_2 = "label_invoice_ner_2.txt"
    input1_path = os.path.join(dataset_dir_path, input_file_1)
    input2_path = os.path.join(dataset_dir_path, input_file_2)
    output_file = "label_invoice_ner_1128.txt"
    output_path = os.path.join(dataset_dir_path, output_file)
    #
    merge_file(input1_path, input2_path, output_path)


if __name__ == "__main__":
    main()
