import os.path
import numpy as np
import os

import os

file1_path = '/data/jiangjiewei/dk_test/features/DenseNet121/val/densenet121'
file2_path = '/data/jiangjiewei/dk_test/features/Inception_v3/val/inception_v3'
output_path = '/data/jiangjiewei/dk_test/features/DenseNe121_inception_v3/val'  # 新文件路径
file_list = ['features_keratitis.txt', 'features_normal.txt', 'features_other.txt']

for file_name in file_list:
    file1 = os.path.join(file1_path, file_name)
    file2 = os.path.join(file2_path, file_name)
    output = os.path.join(output_path, file_name)  # 新文件路径

    with open(file1, 'r') as f1, open(file2, 'r') as f2, open(output, 'w') as out_file:
        lines1 = f1.readlines()  # 读取第一个文件的所有行
        lines2 = f2.readlines()  # 读取第二个文件的所有行

        # 检查两个文件行数是否相同
        if len(lines1) != len(lines2):
            print(f"The number of lines in {file_name} files is not the same.")
        else:
            # 拼接并写入新文件
            for line1, line2 in zip(lines1, lines2):
                line1 = line1.strip()  # 去除换行符
                line2 = line2.strip()  # 去除换行符
                merged_line = f"{line1} {line2}"  # 拼接两行数据
                out_file.write(merged_line + '\n')  # 写入新文件，并换行



# with open('/data/jiangjiewei/dk_test/features/DenseNe121_inception_v3/features_normal.txt', 'r') as file:
#     lines = file.readlines()  # 读取所有行
#
# num_lines = len(lines)  # 获取行数
# num_columns = len(lines[0].strip().split(' '))  # 获取列数（假设以逗号分隔）
#
# print(f"Number of lines: {num_lines}")
# print(f"Number of columns: {num_columns}")
