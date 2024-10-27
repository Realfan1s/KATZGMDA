import numpy as np

precision_list = np.loadtxt('C:\新建文件夹\MKGCN-main\code\LRLSHMDA_precision.txt', delimiter='\t')



with open('C:\新建文件夹\MKGCN-main\code\LRLSHMDA_precision888.txt', 'w') as file:
    for i in range(len(precision_list)):
        file.write(str(1-precision_list[i]) + '\n')