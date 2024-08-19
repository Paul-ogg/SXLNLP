
import csv
import random
from config import Config
# 打开CSV文件
with open('文本分类练习.csv', mode='r', newline='', encoding='utf-8') as file:
    # 创建一个CSV阅读器
    csv_reader = csv.reader(file)

    # 遍历CSV文件的每一行
    rows = [row for row in csv_reader]

    # 计算每行的平均长度
    lengths = [len(row[1]) for row in rows]
    mean_length = sum(lengths) / len(lengths)
    Config["max_length"] = int(mean_length)
    print(f"每行的平均长度为: {mean_length}")

   
# 注意：如果CSV文件包含标题行，你可能需要在遍历之前先读取标题行


