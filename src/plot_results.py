import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

reduce_files = 'results.csv'
labels = 'floyd-warshall-fhe'

os.makedirs('plots', exist_ok=True)	# create directory if not exists
dir_name = 'plots/'

width = 0.3
# REDUCE TIMINGS
fig_reduce, axs_reduce = plt.subplots(nrows=3, ncols=2, figsize=(8,6), constrained_layout=True)
fig_reduce.suptitle('Timings of the Floyd Warshall EVA Program for Various Input Sizes')

# REDUCE FILES

distances_file = reduce_files
# Header: NodeCount,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse
df = pd.read_csv(distances_file).drop(columns=['SimCnt'])
gb = df.groupby(['NodeCount'])
gb_values = list(gb.groups)
mean = gb.mean()
std = gb.std()
x_pos = np.arange(len(gb_values))
axs_reduce[0,0].bar(x_pos, mean['CompileTime'], width, yerr=std['CompileTime'], align='center', capsize=3, label=labels)
axs_reduce[1,0].bar(x_pos, mean['KeyGenerationTime'], width, yerr=std['KeyGenerationTime'], align='center', capsize=3, label=labels)
axs_reduce[2,0].bar(x_pos, mean['EncryptionTime'], width, yerr=std['EncryptionTime'], align='center', capsize=3, label=labels)
axs_reduce[0,1].bar(x_pos, mean['ExecutionTime'], width, yerr=std['ExecutionTime'], align='center', capsize=3, label=labels)
axs_reduce[1,1].bar(x_pos, mean['DecryptionTime'], width, yerr=std['DecryptionTime'], align='center', capsize=3, label=labels)
axs_reduce[2,1].bar(x_pos, mean['ReferenceExecutionTime'], width, yerr=std['ReferenceExecutionTime'], align='center', capsize=3, label=labels)

axs_reduce[0, 0].set_title('Compile Times')
axs_reduce[1, 0].set_title('Key Generation Times')
axs_reduce[2, 0].set_title('Encryption Times')
axs_reduce[0, 1].set_title('Execution Times')
axs_reduce[1, 1].set_title('Decryption Times')
axs_reduce[2, 1].set_title('Reference Execution Times')

for i in range(3):
	for j in range(2):
		axs_reduce[i,j].legend(loc='best')
plt.setp(axs_reduce, xlabel='Graph Node Size', ylabel='Time (ms)', xticks=x_pos, xticklabels=gb_values, xlim=(x_pos[0]-1.5*width, x_pos[-1]+1.5*width))
# plt.show()
plt.savefig(dir_name + 'timings.png')
plt.clf()	# clear the saved figure


# MSE
fig_mse, axs_mse = plt.subplots(nrows=1, ncols=1, figsize=(8,4), constrained_layout=True)
fig_mse.suptitle('MSE of the Floyd Warshall EVA Program for Various Input Sizes')

# REDUCE FILES
reduce_file = reduce_files
# Header: NodeCount,SimCnt,CompileTime,KeyGenerationTime,EncryptionTime,ExecutionTime,DecryptionTime,ReferenceExecutionTime,Mse
df = pd.read_csv(reduce_file).drop(columns=['SimCnt'])
gb = df.groupby(['NodeCount'])
gb_values = list(gb.groups)
mean = gb.mean()
std = gb.std()
x_pos = np.arange(len(gb_values))
axs_mse.bar(x_pos + width*(i-1), mean['Mse'], width, yerr=std['Mse'], align='center', capsize=3, label=labels[i])

axs_mse.set_title('Reduce Ones EVA Program')

# for i in range(2):
# 	axs_mse[i].legend(loc='upper right')
# 	axs_mse[i].set_ylim(bottom=0)
plt.setp(axs_mse, xlabel='Graph Node Size', ylabel='MSE', xticks=x_pos, xticklabels=gb_values, xlim=(x_pos[0]-1.5*width, x_pos[-1]+1.5*width))
plt.savefig(dir_name + 'mse.png')