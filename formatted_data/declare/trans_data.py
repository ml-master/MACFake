import json
import csv
import os
import re

json_files = '/home/amax/sunyishanProject/ECAL2021/formatted_data/declare/gossicop_v3-3/gossipcop_v3-7_integration_based_legitimate_tn300.json'
outroot = '/home/amax/sunyishanProject/ECAL2021/formatted_data/declare/Snopes'
train_name = 'train_0.tsv'
test_name = 'test_0.tsv'

outpath = os.path.join(outroot, train_name)

# 读取JSON文件
with open(json_files, 'r') as json_file:
    data = json.load(json_file)

# 打开一个TSV文件进行写入
with open(outpath, 'w', newline='', encoding='utf-8') as tsv_file:
    writer = csv.writer(tsv_file, delimiter='\t')

    # 写入TSV文件的表头
    writer.writerow(
        ['id_left', 'cred_label', 'claim_id', 'claim_text', 'claim_source', 'id_right', 'evidence', 'evidence_source'])


    def process_text(text):
        if isinstance(text, str):
            # 将转义字符进行处理
            text = text.replace("\n", "\\n").replace("\t", "\\t")
        return text


    def extract_numbers(text):
        return ''.join(re.findall(r'\d+', text))


    # 遍历JSON数据并提取字段写入TSV文件
    id_left_number = 2696
    for key, value in data.items():
        id_left_number += 1
        id_left = str(id_left_number)
        cred_label = 'true'
        claim_id = key
        claim_text = process_text(value['generated_text_t01'])
        claim_source = ' '
        evidence_source = ' '

        id_right_1 = extract_numbers(value['doc_1_id'])
        evidence_1 = process_text(value['doc_1_text'])
        # 写入一行到TSV文件
        writer.writerow([id_left, cred_label, claim_id, claim_text , claim_source,  id_right_1, evidence_1, evidence_source])

        id_right_2 = extract_numbers(value['doc_2_id'])
        evidence_2 = process_text(value['doc_2_text'])
        writer.writerow([id_left, cred_label, claim_id, claim_text, claim_source, id_right_2, evidence_2, evidence_source])


