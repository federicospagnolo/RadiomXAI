################
#written by Federico Spagnolo
#usage: python bootstrap_radiomics_classifier.py
################

import numpy as np
import pandas as pd
import os, sys
import openpyxl
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Use a non-interactive backend
matplotlib.use('Agg')

script_folder = os.path.dirname(os.path.realpath(__file__))

# Load training dataframe
train_csv_filename = os.path.join(script_folder, "train_features_5.csv")
df = pd.read_csv(train_csv_filename, index_col=None)

X = df.iloc[:, 1:].values  # features
y = df.iloc[:, 0].values
y = np.where(y == 'TP', 1, 0)  # Binary target
feature_names = df.columns[1:].tolist()  # feature names

# Standardizing the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initializing and training the logistic regression model
model = LogisticRegression(max_iter=10000, solver='liblinear', C=1.0, penalty='l1', class_weight = {0:0.29, 1:0.71})

# Getting the feature importance (coefficients)
model.fit(X_scaled, y)
feature_importance = model.coef_[0] / np.max(np.abs(model.coef_[0]))

# Add this line to pair feature names with their importance
feature_importance_pairs = list(zip(feature_names, feature_importance))

# Sort the feature_importance_pairs by absolute importance
sorted_pairs = sorted(feature_importance_pairs, key=lambda x: abs(x[1]), reverse=True)

# Split the sorted pairs into positive and negative importance
positive_pairs = [pair for pair in sorted_pairs if pair[1] > 0]
negative_pairs = [pair for pair in sorted_pairs if pair[1] < 0]

# Define the number of features per subplot
features_per_subplot = 10

# Splitting sorted feature names and importances into two parts for two subplots
feature_names_split_pos = [pair[0] for pair in positive_pairs[:features_per_subplot]][::-1]  # Reverse the order
importance_split_pos = [pair[1] for pair in positive_pairs[:features_per_subplot]][::-1]  # Reverse the order
feature_names_split_neg = [pair[0] for pair in negative_pairs[:features_per_subplot]]
importance_split_neg = [pair[1] for pair in negative_pairs[:features_per_subplot]]

# Plotting feature importance in specific subplots (0,0) and (0,1)
plot_filename = os.path.join(script_folder, "train_feature_5_importance.png")
fig, axs = plt.subplots(1, 2, figsize=(18, 14), sharex=True)  # Create a 1x2 grid

# Plotting positive coefficients on the top of the first subplot
bars_pos = axs[0].barh(feature_names_split_pos, importance_split_pos, color='skyblue')
axs[0].set_xlabel('Coefficient Value', fontsize=16, fontweight='bold')
axs[0].set_ylabel('', fontsize=16, fontweight='bold')
axs[0].axvline(x=.3, color='r', linestyle='--')
axs[0].axvline(x=0, linestyle='-')
axs[0].tick_params(axis='y', labelsize=16)
axs[0].tick_params(axis='x', labelsize=16, labelbottom=True)
axs[0].grid(True)
axs[0].set_xticks([-1, 0, 1])
plt.setp(axs[0].get_yticklabels(), fontweight='bold')
plt.setp(axs[0].get_xticklabels(), fontweight='bold')

# Adding coefficient values next to the bars in the first subplot
for bar in bars_pos:
    width = bar.get_width()
    label_x_pos = width if width < 0 else width + 0.1
    axs[0].text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center', fontsize=15, fontweight='bold')

# Plotting negative coefficients on the bottom of the second subplot
bars_neg = axs[1].barh(feature_names_split_neg, importance_split_neg, color='skyblue')
axs[1].set_xlabel('Coefficient Value', fontsize=16, fontweight='bold')
axs[1].set_ylabel('', fontsize=16, fontweight='bold')
axs[1].axvline(x=-.3, color='g', linestyle='--')
axs[1].axvline(x=0, linestyle='-')
axs[1].tick_params(axis='y', labelsize=16)
axs[1].tick_params(axis='x', labelsize=16, labelbottom=True)
axs[1].grid(True)
axs[1].set_xticks([-1, 0, 1])
plt.setp(axs[1].get_yticklabels(), fontweight='bold')
plt.setp(axs[1].get_xticklabels(), fontweight='bold')

# Adding coefficient values next to the bars in the second subplot
for bar in bars_neg:
    width = bar.get_width()
    label_x_pos = width if width < 0 else width + 0.1
    axs[1].text(label_x_pos, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', va='center', fontsize=15, fontweight='bold')

fig.suptitle('Feature Importance in LR', fontsize=20, fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(plot_filename)
sys.exit()
# Load test dataframe
test_csv_filename = os.path.join(script_folder, "test_features_5.csv")
test_df = pd.read_csv(test_csv_filename, index_col=None)

X_new = test_df.iloc[:, 1:].values  # features
y_new = test_df.iloc[:, 0].values
y_new = np.where(y_new == 'TP', 1, 0)  # Binary target
feature_names = test_df.columns[1:].tolist()  # feature names

# Standardizing the features
X_new_scaled = scaler.fit_transform(X_new)

# Number of bootstrap samples
n_iterations = 1000
f1_scores = []
ppv_scores = []

# Bootstrapping
print('Bootstrapping iterations...')
for i in tqdm(range(n_iterations)):
    # Create a bootstrap sample from the test set
    X_resampled, y_resampled = resample(X_new_scaled, y_new)
    
    # Make predictions on the resampled test set
    y_pred_resampled = model.predict(X_resampled)
    
    # Compute F1 score
    conf_matrix = confusion_matrix(y_resampled, y_pred_resampled)
    precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
    recall = conf_matrix[1, 1] / (789 + conf_matrix[1, 1] + conf_matrix[1, 0])
    f1_scores.append(2*precision*recall/(precision+recall))
    ppv_scores.append(precision)

# Convert to a NumPy array for convenience
f1_scores = np.array(f1_scores)
ppv_scores = np.array(ppv_scores)

# Calculate the confidence intervals
alpha = 0.95
lowerf = np.percentile(f1_scores, (1.0 - alpha) / 2.0 * 100)
upperf = np.percentile(f1_scores, (1.0 + alpha) / 2.0 * 100)
lowerp = np.percentile(ppv_scores, (1.0 - alpha) / 2.0 * 100)
upperp = np.percentile(ppv_scores, (1.0 + alpha) / 2.0 * 100)

# Making predictions on the new dataset
predictions = model.predict(X_new_scaled)

conf_matrix = confusion_matrix(y_new, predictions)

precision = 3050 / (3050 + 1818)
recall = 3050 / (3050 + 789)

F1 = 2 * precision * recall / (precision + recall)

new_precision = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[0, 1])
new_recall = conf_matrix[1, 1] / (789 + conf_matrix[1, 1] + conf_matrix[1, 0])

new_F1 = 2*new_precision*new_recall/(new_precision + new_recall)

# Save precision, recall, F1 score, and confidence interval to XLSX
wb = openpyxl.Workbook()
ws = wb.active

# Write precision and recall to the worksheet
ws.append(['F1', F1])
ws.append(['New F1', new_F1])
ws.append(['PPV', precision])
ws.append(['New PPV', new_precision])
ws.append(['Bootstrap F1', np.mean(f1_scores)])
ws.append(['F1 Score Confidence Interval', f'[{lowerf:.4f}, {upperf:.4f}]'])
ws.append(['Bootstrap PPV', np.mean(ppv_scores)])
ws.append(['PPV Confidence Interval', f'[{lowerp:.4f}, {upperp:.4f}]'])
ws.append(['Confusion Matrix', 'Predicted Negative', 'Predicted Positive'])
ws.append(['Actual Negative', conf_matrix[0, 0], conf_matrix[0, 1]])
ws.append(['Actual Positive', 789 + conf_matrix[1, 0], conf_matrix[1, 1]])

# Save the workbook
wb.save('final_metrics_5.xlsx')
