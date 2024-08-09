################
#written by Federico Spagnolo
#usage: python compare_features.py
################

import numpy as np
import os
import sys
import glob
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

script_folder = os.path.dirname(os.path.realpath(__file__))

# Load training dataframe
train_csv_filename = os.path.join(script_folder, "train_features_5.csv")
df = pd.read_csv(train_csv_filename, index_col=None)
# Load test dataframe
test_csv_filename = os.path.join(script_folder, "test_features_5.csv")
test_df = pd.read_csv(test_csv_filename, index_col=None)

# Filter the DataFrame to include only rows where group is "TP"
train_tp_rows = df[df['group'] == 'TP']
train_fp_rows = df[df['group'] == 'FP']
test_tp_rows = test_df[test_df['group'] == 'TP']
test_fp_rows = test_df[test_df['group'] == 'FP']

# Calculate the mean of the max/min/mean column for these rows
mean_max_train_tp = train_tp_rows['original_firstorder_Maximum'].mean()
std_max_train_tp = train_tp_rows['original_firstorder_Maximum'].std()
mean_max_test_tp = test_tp_rows['original_firstorder_Maximum'].mean()
std_max_test_tp = test_tp_rows['original_firstorder_Maximum'].std()
max_p = mannwhitneyu(train_tp_rows['original_firstorder_Maximum'], test_tp_rows['original_firstorder_Maximum'])[1]

mean_min_train_tp = train_tp_rows['original_firstorder_Minimum'].mean()
std_min_train_tp = train_tp_rows['original_firstorder_Minimum'].std()
mean_min_test_tp = test_tp_rows['original_firstorder_Minimum'].mean()
std_min_test_tp = test_tp_rows['original_firstorder_Minimum'].std()
min_p = mannwhitneyu(train_tp_rows['original_firstorder_Minimum'], test_tp_rows['original_firstorder_Minimum'])[1]

mean_mean_train_tp = train_tp_rows['original_firstorder_Mean'].mean()
std_mean_train_tp = train_tp_rows['original_firstorder_Mean'].std()
mean_mean_test_tp = test_tp_rows['original_firstorder_Mean'].mean()
std_mean_test_tp = test_tp_rows['original_firstorder_Mean'].std()
mean_p = mannwhitneyu(train_tp_rows['original_firstorder_Mean'], test_tp_rows['original_firstorder_Mean'])[1]

with open('compare_features.txt', 'w') as f:
	sys.stdout = f
	print('True positive examples...')
	print(f'Max in training set: {mean_max_train_tp} ± {std_max_train_tp}')
	print(f'Max in test set: {mean_max_test_tp} ± {std_max_test_tp}')

	print(f'Min in training set: {mean_min_train_tp} ± {std_min_train_tp}')
	print(f'Min in test set: {mean_min_test_tp} ± {std_min_test_tp}')

	print(f'Mean in training set: {mean_mean_train_tp} ± {std_mean_train_tp}')
	print(f'Mean in test set: {mean_mean_test_tp} ± {std_mean_test_tp}')

	print(f'Max p-value: {max_p}')
	print(f'Mean p-value: {mean_p}')
	print(f'Min p-value: {min_p}')

# Set up the plot
categories = ['Max', 'Mean', 'Min']
x = np.arange(len(categories))

fig, ax = plt.subplots()

# Plot training set
ax.errorbar(x, [mean_max_train_tp, mean_mean_train_tp, mean_min_train_tp], yerr=[std_max_train_tp, std_mean_train_tp, std_min_train_tp], label='Training Set', capsize=5, color='skyblue', fmt='o', linestyle='none')

# Plot test set
ax.errorbar(x, [mean_max_test_tp, mean_mean_test_tp, mean_min_test_tp], yerr=[std_max_test_tp, std_mean_test_tp, std_min_test_tp], label='Test Set', capsize=5, color='orange', fmt='o', linestyle='none')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Mean and Standard Deviation of TPs', fontsize=14, fontweight='bold')
ax.set_ylabel('Saliency', fontsize=14, fontweight='bold')
#ax.set_title('Mean and Standard Deviation of TPs in Training and Test Sets', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--')

# Show the plot
plt.savefig('compareTP.png', dpi=800)

mean_max_train_fp = train_fp_rows['original_firstorder_Maximum'].mean()
std_max_train_fp = train_fp_rows['original_firstorder_Maximum'].std()
mean_max_test_fp = test_fp_rows['original_firstorder_Maximum'].mean()
std_max_test_fp = test_fp_rows['original_firstorder_Maximum'].std()
max_p = mannwhitneyu(train_fp_rows['original_firstorder_Maximum'], test_tp_rows['original_firstorder_Maximum'])[1]

mean_min_train_fp = train_fp_rows['original_firstorder_Minimum'].mean()
std_min_train_fp = train_fp_rows['original_firstorder_Minimum'].std()
mean_min_test_fp = test_fp_rows['original_firstorder_Minimum'].mean()
std_min_test_fp = test_fp_rows['original_firstorder_Minimum'].std()
min_p = mannwhitneyu(train_fp_rows['original_firstorder_Minimum'], test_tp_rows['original_firstorder_Minimum'])[1]

mean_mean_train_fp = train_fp_rows['original_firstorder_Mean'].mean()
std_mean_train_fp = train_fp_rows['original_firstorder_Mean'].std()
mean_mean_test_fp = test_fp_rows['original_firstorder_Mean'].mean()
std_mean_test_fp = test_fp_rows['original_firstorder_Mean'].std()
mean_p = mannwhitneyu(train_fp_rows['original_firstorder_Mean'], test_tp_rows['original_firstorder_Mean'])[1]


with open('compare_features.txt', 'a') as f:
	sys.stdout = f
	print('False positive examples...')
	print(f'Max in training set: {mean_max_train_fp} ± {std_max_train_fp}')
	print(f'Max in test set: {mean_max_test_fp} ± {std_max_test_fp}')

	print(f'Mean in training set: {mean_min_train_fp} ± {std_min_train_fp}')
	print(f'Mean in test set: {mean_min_test_fp} ± {std_min_test_fp}')

	print(f'Min in training set: {mean_mean_train_fp} ± {std_mean_train_fp}')
	print(f'Min in test set: {mean_mean_test_fp} ± {std_mean_test_fp}')

	print(f'Max p-value: {max_p}')
	print(f'Mean p-value: {mean_p}')
	print(f'Min p-value: {min_p}')
sys.stdout = sys.__stdout__	

fig, ax = plt.subplots()

# Plot training set
ax.errorbar(x, [mean_max_train_fp, mean_mean_train_fp, mean_min_train_fp], yerr=[std_max_train_fp, std_mean_train_fp, std_min_train_fp], label='Training Set', capsize=5, color='skyblue', fmt='o', linestyle='none')

# Plot test set
ax.errorbar(x, [mean_max_test_fp, mean_mean_test_fp, mean_min_test_fp], yerr=[std_max_test_fp, std_mean_test_fp, std_min_test_fp], label='Test Set', capsize=5, color='orange', fmt='o', linestyle='none')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('Mean and Standard Deviation of FPs', fontsize=14, fontweight='bold')
ax.set_ylabel('Saliency', fontsize=14, fontweight='bold')
#ax.set_title('Mean and Standard Deviation of FPs in Training and Test Sets', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax.legend(fontsize=12)
ax.grid(True, linestyle='--')

# Show the plot
plt.savefig('compareFP.png', dpi=800)
