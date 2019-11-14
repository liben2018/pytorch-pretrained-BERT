import os
import sys
import json


def make_new_path(directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_classname_dict(file_path):
    eng_start = 0
    eng_end = 0
    ja_start = 0
    # ja_end = 0
    invoice_class_dict = dict()
    with open(file_path, 'r') as f:
        lines = f.readlines()
        lines = [x.strip() for x in lines if x not in ['\n', '']]
        # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）或字符序列。
        # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        # 如果你想摆脱一切“伪造”，例如空字符串，空元组，零，你也可以使用
        # list2 = [x for x in list1 if x]
        lines = ["".join(x.split()) for x in lines]
        # "".join(s.split())
        for i, line in enumerate(lines):
            if line == "English:":
                eng_start = i+1
            if line == "日本語:":
                eng_end = i-1
                ja_start = i+1
        for i, line in enumerate(lines):
            if (i >= eng_start) and (i <= eng_end):
                invoice_class_dict[lines[ja_start + i - eng_start]] = line
    ja_list = list(invoice_class_dict.keys())
    eng_list = list(invoice_class_dict.values())
    assert len(ja_list) == len(eng_list)
    return invoice_class_dict, ja_list, eng_list


def build_map_string(text):
    map_char = []
    for i, char in enumerate(text):
        map_char.append(char)
        map_char[i] = char
    return map_char


def ja_class_name_adjust(label):
    """
    Adjust a mismatched label to match ja_class_name_list!
    """
    if label == '当月買上消費税_v':
        label = '当月買上額消費税_v'
    if label == '今回請求額_v':
        label = '今回請求金額_v'
    if label == '前回請求額_v':
        label = '前回請求金額_v'
    return label


def create_ner_datasets(input_file_path, output_file_dir_path, class_name_file_path, label_file_path):
    invoice_class_dict, ja_list, eng_list = get_classname_dict(class_name_file_path)
    out_of_list = []
    output_file_line_list = []
    with open(input_file_path, 'r') as f:
        # data = json.load(f) # cannot be load because of too many lines of dict in files!
        dict_lines = f.readlines()

        # Treat one invoice data!
        for i, dict_line in enumerate(dict_lines):
            line_output_print = []
            dict_line = json.loads(dict_line)
            text = dict_line["text"]
            maps = build_map_string(text)
            labels = dict_line["labels"]
            invoice_file = text.split("\t")[0]
            # type(labels): list
            for label in labels:
                # print("original text:", text[label[0]:label[1]], label[2])
                # print("char-bas list:", maps[label[0]:label[1]], label[2])
                label[2] = ja_class_name_adjust(label[2])
                if label[2] not in ja_list:
                    print(label, "not exist in ja_list!")
                    out_of_list.append(label[2])

            find_out_class_name = True
            if not find_out_class_name:
                assert (len(out_of_list) == 0), "Need to find out all of ja_class_name in annotated file!"

            for j, char in enumerate(text):
                ner_label = "O"
                print_line = " ".join([text[j], ner_label, invoice_file])
                line_output_print.append(print_line)

            for label in labels:
                label[2] = invoice_class_dict[label[2]]
                text_data = text[label[0]:label[1]]
                # text_data = text[label[0]:label[1]].strip()
                use_IOB = False
                use_IOBES = True
                if use_IOBES:
                    if len(text_data) == 1:
                        ner_label = "S-"+label[2]
                        print_line = " ".join([text_data, ner_label, invoice_file])
                        line_output_print[label[0]] = print_line
                    if len(text_data) > 1:
                        for k, char_in_word in enumerate(text_data):
                            if k == 0:
                                ner_label = "B-"+label[2]
                                print_line = " ".join([char_in_word, ner_label, invoice_file])
                                line_output_print[label[0] + k] = print_line
                            if k == len(text_data)-1:
                                ner_label = "E-"+label[2]
                                print_line = " ".join([char_in_word, ner_label, invoice_file])
                                line_output_print[label[0] + k] = print_line
                            if k < (len(text_data)-1) and (k > 0):
                                ner_label = "I-"+label[2]
                                print_line = " ".join([char_in_word, ner_label, invoice_file])
                                line_output_print[label[0] + k] = print_line
                elif use_IOB:
                    assert len(text_data) >= 1
                    for k, char_in_word in enumerate(text_data):
                        if k == 0:
                            ner_label = "B-"+label[2]
                            print_line = " ".join([char_in_word, ner_label, invoice_file])
                            line_output_print[label[0] + k] = print_line
                        elif k <= (len(text_data)-1) and (k > 0):
                            ner_label = "I-"+label[2]
                            print_line = " ".join([char_in_word, ner_label, invoice_file])
                            line_output_print[label[0] + k] = print_line
                        else:
                            print("k<0 or k>=len(text_data) item in use_IOB module!")
                else:
                    print("Please tell me which of IOB and IOBES do you want to use!")
            line_output_print.append('\n')  # split two line by ' '
            output_file_line_list.extend(line_output_print)
            # list_a.extend(list_b): [list_a_1, list_b_1]
            # list_a.append(list_b): [list_a_1, [list_b]]
    # output_file_line_list = [x for x in output_file_line_list if x[0] not in [' ']]
    new_list = []
    for x in output_file_line_list:
        if x[0] in ['\t']:
            continue
        if x == '\n':
            new_list.append(x)
        elif len(x.strip().split()) == 3:
            new_list.append(x)
    output_file_line_list = new_list

    # write label file
    label_list = []
    for line in output_file_line_list:
        raw_line = line.strip().split(' ')
        if len(raw_line) == 3:
            label = raw_line[1]
            if label not in label_list:
                label_list.append(label)
    with open(label_file_path, 'w') as f:
        for label in label_list:
            print('label:', label)
            if label in ['\n', '\t']:
                pass
            else:
                f.write("%s\n" % label)

    # x.strip()
    total_data = len(output_file_line_list)
    list_train = output_file_line_list[0:int(total_data*0.8)]
    list_dev = output_file_line_list[int(total_data*0.8):int(total_data*0.9)]
    list_test = output_file_line_list[int(total_data*0.9):]
    assert total_data == len(list_test) + len(list_dev) + len(list_train)

    train_file_path = os.path.join(output_file_dir_path, 'train.txt')
    dev_file_path = os.path.join(output_file_dir_path, 'dev.txt')
    test_file_path = os.path.join(output_file_dir_path, 'test.txt')

    with open(train_file_path, 'w') as f:
        for item in list_train:
            if item == '\n':
                f.write("%s" % item)
            else:
                f.write("%s\n" % item)
    with open(dev_file_path, 'w') as f:
        for item in list_dev:
            if item == '\n':
                f.write("%s" % item)
            else:
                f.write("%s\n" % item)
    with open(test_file_path, 'w') as f:
        for item in list_test:
            if item == '\n':
                f.write("%s" % item)
            else:
                f.write("%s\n" % item)
    # create_split_datasets(text, labels, dataset_files_path)


def main():
    PATH_DIR = os.getcwd()
    print(PATH_DIR)
    #
    input_file = "export_20191106.json"
    raw_date_files_path = os.path.join(PATH_DIR, input_file)
    #
    output_dataset_dir = 'NER_data_1114_IOBES_Transformers'
    dataset_dir_path = os.path.join(PATH_DIR, output_dataset_dir)
    make_new_path(dataset_dir_path)
    #
    class_name_file = 'class_name_invoice.txt'
    class_name_file_path = os.path.join(PATH_DIR, class_name_file)
    label_file = 'label_invoice_ner.txt'
    label_file_path = os.path.join(dataset_dir_path, label_file)
    create_ner_datasets(raw_date_files_path, dataset_dir_path, class_name_file_path, label_file_path)


if __name__ == "__main__":
    main()

"""
sys.exit()

final_list.append(str.join(seq))
final_list = []
for file in os.listdir(files_path):
    str = '\t'
    with open(file) as f:
        seq = f.readlines()
    final_list.append(str.join(seq))
"""