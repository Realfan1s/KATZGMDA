# 打开包含逗号的字符串数据文件
with open('drug_mic_matrix9.txt', 'r') as file:
    # 读取文件内容，并将每一行以逗号分割成字符串列表
    lines = [line.strip().split(' ') for line in file]

# 将包含逗号的字符串列表转换为浮点数列表
float_lines = [[float(value) for value in line] for line in lines]

# 将转换后的浮点数列表写入新的文本文件，保持原始的行列结构
with open('drug_mic_matrix10.txt', 'w') as file:
    for line in float_lines:
        file.write(','.join(map(str, line)) + '\n')