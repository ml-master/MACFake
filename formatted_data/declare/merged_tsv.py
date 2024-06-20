import pandas as pd
import os

output_root = '/home/amax/sunyishanProject/ECAL2021/formatted_data/declare/Snopes/mapped_data/1fold'
input_root = '/home/amax/sunyishanProject/ECAL2021/formatted_data/declare'

train_name_false = 'train_false_data.tsv'
test_name_false = 'test_false_data.tsv'
validation_name_false ='validation_false_data.tsv'

train_name_true = 'train_true_data.tsv'
test_name_true = 'test_true_data.tsv'
validation_name_true ='validation_true_data.tsv'


def megerd(path1,path2,out):
    # 读取两个 TSV 文件
    df1 = pd.read_csv(path1, sep='\t')
    df2 = pd.read_csv(path2, sep='\t')

    # 合并两个 DataFrame
    merged_df = pd.concat([df1, df2], ignore_index=True)

    # 将合并后的 DataFrame 保存为一个新的 TSV 文件
    merged_df.to_csv(out, sep='\t', index=False)

input_train_false = os.path.join(input_root, train_name_false)
input_train_true = os.path.join(input_root, train_name_true)
output_train_path = os.path.join(output_root, 'train_0.tsv')
megerd(input_train_false, input_train_true, output_train_path)

input_test_false = os.path.join(input_root, test_name_false)
input_test_true = os.path.join(input_root, test_name_true)
output_test_path = os.path.join(output_root, 'test_0.tsv')
megerd(input_test_false, input_test_true, output_test_path)

input_validation_false = os.path.join(input_root, validation_name_false)
input_validation_true = os.path.join(input_root, validation_name_true)
output_validation_path = os.path.join(output_root, 'validation_0.tsv')
megerd(input_validation_false, input_validation_true, output_validation_path)