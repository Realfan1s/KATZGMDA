import numpy as np
import scipy.sparse as sp





drug_sim = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\DrugVirus\drugsimilarity.txt')
mic_sim = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\DrugVirus\microbesimilarity.txt')


adj_triple = np.loadtxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\DrugVirus\\adj.txt')
drug_mic_matrix = sp.csc_matrix((adj_triple[:, 2], (adj_triple[:, 0] - 1, adj_triple[:, 1] - 1)),
                                    shape=(len(drug_sim), len(mic_sim))).toarray()

np.savetxt('C:\新建文件夹\MKGCN-main\data\MicrobeDrugA\DrugVirus\drug_mic_matrix8.txt',drug_mic_matrix)