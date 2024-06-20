import json
import random

# 读取 JSON 文件
with open('/home/amax/sunyishanProject/ECAL2021/formatted_data/declare/gossicop_v3-3/gossipcop_v3-7_integration_based_legitimate_tn300.json', 'r') as f:
    data = json.load(f)

# 获取数据的总长度
total_length = len(data)

# 计算每一份数据的长度
train_size = int(0.6 * total_length)
validation_size = int(0.2 * total_length)
test_size = total_length - train_size - validation_size


data_items = list(data.items())

# 分割数据
train_data = dict(data_items[:train_size])
validation_data = dict(data_items[train_size:train_size + validation_size])
test_data = dict(data_items[train_size + validation_size:])

# 将分割后的数据写入三个新的 JSON 文件
with open('train_true_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('validation_true_data.json', 'w') as f:
    json.dump(validation_data, f, indent=4)

with open('test_true_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)
