# 打开包含逗号的字符串数据文件
with open('drug_mic_matrix10.txt', 'r') as file:
    # 读取文件内容，并将每一行以逗号分割成字符串列表
    lines = [line.strip().replace(',', ' ') for line in file]

# 将包含空格的字符串列表写入新的文本文件
with open('drug_mic_matrix10.txt', 'w') as file:
    for line in lines:
        file.write(line + '\n')