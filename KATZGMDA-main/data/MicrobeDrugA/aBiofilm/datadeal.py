# 读取科学计数法表示的txt文件
with open('adj_matrix.txt', 'r') as file:
    data = file.readlines()

# 将科学计数法转换为浮点计数形式
converted_data = []
for line in data:
    parts = line.split()  # 假设数据中使用空格分隔
    converted_parts = [format(float(x), '.10f') for x in parts]  # 将每个部分从科学计数法转换为浮点数
    converted_line = '\t'.join(converted_parts)  # 以制表符分隔各部分
    converted_data.append(converted_line)

# 将结果写入新文件
with open('drug_mic_matrix8.txt', 'w') as file:
    file.write('\n'.join(converted_data))