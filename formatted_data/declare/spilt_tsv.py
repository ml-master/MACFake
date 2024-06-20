import pandas as pd

# 读取TSV文件，包括表头
df = pd.read_csv('/home/amax/sunyishanProject/ECAL2021/formatted_data/declare/Snopes/train_0.tsv', sep='\t')

# 获取总行数（不包括表头）
total_rows = len(df)

# 计算每份应该有多少行（包括表头）
# 注意：这里假设表头只占一行，因此总行数-1是数据行的数量
# 每份文件的行数按照6:2:2的比例分配（不包括表头）
rows_per_part = [(total_rows - 1) // 10 * 6 + 1, (total_rows - 1) // 10 * 2 + 1, (total_rows - 1) // 10 * 2 + 1]

# 确保总行数匹配

# 初始化起始索引
start_idx = 0

# 遍历每份文件
for i, rows in enumerate(rows_per_part):
    # 计算结束索引（不包括）
    end_idx = start_idx + rows - 1
    # 如果不是最后一份文件，并且数据行不够分配，则取到剩余所有行
    if i < len(rows_per_part) - 1 and end_idx > total_rows:
        end_idx = total_rows

        # 获取当前份的数据（包括表头）
    part_df = df.iloc[start_idx:end_idx]

    # 将当前份的数据写入新的TSV文件
    part_df.to_csv(f'part_{i + 1}.tsv', sep='\t', index=False)

    # 更新起始索引为下一份的起始位置
    start_idx = end_idx

# 注意：如果总行数不能被10整除，上述代码可能会导致某些部分的行数稍微多一些或少一些
# 为了确保严格的比例，可能需要更复杂的逻辑来处理边界情况