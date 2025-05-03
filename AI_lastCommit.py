# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# from sklearn.model_selection import train_test_split
# from sklearn.impute import SimpleImputer
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Step 1: Load and Combine the Data
# batches = []
# for i in range(1, 11):
#     batch = pd.read_csv(f'batch{i}.dat', sep=' ', header=None)
#     batches.append(batch)

# data = pd.concat(batches, ignore_index=True)

# data[0] = data[0].astype(str)
# y = data[0].apply(lambda x: int(x.split(';')[0]))
# concentrations = data[0].apply(lambda x: float(x.split(';')[1]))
# concentrations_secondary = data[1].apply(lambda x: float(x.split(':')[1]))

# feature_columns = data.iloc[:, 2:130]
# X = np.zeros((data.shape[0], 128))
# for col_idx in range(128):
#     column = feature_columns.iloc[:, col_idx].astype(str)
#     X[:, col_idx] = column.apply(lambda x: float(x.split(':')[1]) if ':' in x else float(x) if x != 'nan' else np.nan)

# imputer = SimpleImputer(strategy='mean')
# X = imputer.fit_transform(X)

# print(f"X shape: {X.shape}")
# print(f"y shape: {y.shape}")
# print(f"Concentrations shape: {concentrations.shape}")
# print("Class distribution:\n", pd.Series(y).value_counts())
# print("First few rows of X:\n", X[:5, :5])
# print(f"Total NaN values in X after imputation: {np.isnan(X).sum()}")

# # Step 2.1: Feature Scaling
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# print("Mean of scaled features:", X_scaled.mean(axis=0).round(5))
# print("Std of scaled features:", X_scaled.std(axis=0).round(5))

# # Step 2.2: Dimensionality Reduction (PCA)
# pca = PCA(n_components=10)
# X_pca = pca.fit_transform(X_scaled)
# print(f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_}")
# print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# # Step 2.3: Train-Test Split
# X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
# X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
# print(f"Training set size (PCA): {X_train_pca.shape[0]}")
# print(f"Test set size (PCA): {X_test_pca.shape[0]}")
# print("Training class distribution:\n", pd.Series(y_train).value_counts())
# print("Test class distribution:\n", pd.Series(y_test).value_counts())

# # Step 3.1: KNN Classification
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train_pca, y_train)
# y_pred_knn = knn.predict(X_test_pca)

# knn_accuracy = accuracy_score(y_test, y_pred_knn)
# knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
# knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
# knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

# print("KNN Results:")
# print(f"Accuracy: {knn_accuracy:.4f}")
# print(f"Precision: {knn_precision:.4f}")
# print(f"Recall: {knn_recall:.4f}")
# print(f"F1 Score: {knn_f1:.4f}")

# # Step 3.2: Naïve Bayes Classification
# nb = GaussianNB()
# nb.fit(X_train_scaled, y_train)
# y_pred_nb = nb.predict(X_test_scaled)

# nb_accuracy = accuracy_score(y_test, y_pred_nb)
# nb_precision = precision_score(y_test, y_pred_nb, average='weighted')
# nb_recall = recall_score(y_test, y_pred_nb, average='weighted')
# nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')

# print("Naïve Bayes Results:")
# print(f"Accuracy: {nb_accuracy:.4f}")
# print(f"Precision: {nb_precision:.4f}")
# print(f"Recall: {nb_recall:.4f}")
# print(f"F1 Score: {nb_f1:.4f}")

# # Step 3.3: Random Forest Classification
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train_scaled, y_train)
# y_pred_rf = rf.predict(X_test_scaled)

# rf_accuracy = accuracy_score(y_test, y_pred_rf)
# rf_precision = precision_score(y_test, y_pred_rf, average='weighted')
# rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
# rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

# print("Random Forest Results:")
# print(f"Accuracy: {rf_accuracy:.4f}")
# print(f"Precision: {rf_precision:.4f}")
# print(f"Recall: {rf_recall:.4f}")
# print(f"F1 Score: {rf_f1:.4f}")

# # Step 3.4: K-Means Clustering
# kmeans = KMeans(n_clusters=6, random_state=42)
# kmeans.fit(X_train_pca)
# y_pred_kmeans = kmeans.predict(X_test_pca)

# kmeans_ari = adjusted_rand_score(y_test, y_pred_kmeans)

# print("K-Means Results:")
# print(f"Adjusted Rand Index: {kmeans_ari:.4f}")

# # Step 4: Visualization and Analysis
# # 4.1 PCA Scatter Plot
# gas_names = {1: 'Ethanol', 2: 'Ethylene', 3: 'Ammonia', 4: 'Acetaldehyde', 5: 'Acetone', 6: 'Toluene'}

# plt.figure(figsize=(10, 6))
# for gas_label in range(1, 7):
#     mask = (y == gas_label) # y is gas label from data
#     plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=gas_names[gas_label], alpha=0.6)

# plt.xlabel('PCA Component 1')
# plt.ylabel('PCA Component 2')
# plt.title('PCA Scatter Plot of Gas Data')
# plt.legend()
# plt.grid(True)
# plt.savefig("PCA_Scatter_Plot.png")
# plt.show()

# # 4.2 Confusion Matrix for Random Forest
# cm = confusion_matrix(y_test, y_pred_rf)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(gas_names.values()), yticklabels=list(gas_names.values()))
# plt.xlabel('Predicted')
# plt.ylabel('True')
# plt.title('Confusion Matrix for Random Forest')
# plt.savefig("ConfusionMatrixforRandomForest.png")
# plt.show()

# # 4.3 Bar Chart of Algorithm Performance
# algorithms = ['KNN', 'Naïve Bayes', 'Random Forest']
# accuracies = [knn_accuracy, nb_accuracy, rf_accuracy]

# plt.figure(figsize=(8, 6))
# plt.bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
# plt.xlabel('Algorithm')
# plt.ylabel('Accuracy')
# plt.title('Comparison of Classification Algorithms')
# plt.ylim(0, 1)
# for i, v in enumerate(accuracies):
#     plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
# plt.savefig("Bar_Chart_of_Algorithm_Performance.png")
# plt.show()
# Import necessary libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score, confusion_matrix, silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import time



# Step 1: Load and Combine the Data
print("\n=== Step 1: Loading and Combining Data ===\n")
batches = []
for i in range(1, 11):
    batch = pd.read_csv(f'batch{i}.dat', sep=' ', header=None)
    batches.append(batch)

data = pd.concat(batches, ignore_index=True)
print(f"Total number of instances loaded: {data.shape[0]}")
print(f"Total number of columns (including labels and features): {data.shape[1]}\n")

data[0] = data[0].astype(str)
y = data[0].apply(lambda x: int(x.split(';')[0]))
concentrations = data[0].apply(lambda x: float(x.split(';')[1]))
concentrations_secondary = data[1].apply(lambda x: float(x.split(':')[1]))

feature_columns = data.iloc[:, 2:130]
X = np.zeros((data.shape[0], 128))
for col_idx in range(128):
    column = feature_columns.iloc[:, col_idx].astype(str)
    X[:, col_idx] = column.apply(lambda x: float(x.split(':')[1]) if ':' in x else float(x) if x != 'nan' else np.nan)

print("Feature Statistics Before Imputation:")
missing_values_per_feature = np.sum(np.isnan(X), axis=0)
feature_means_before_scaling = np.nanmean(X, axis=0)
print(f"Number of missing values per feature: {missing_values_per_feature}")
print(f"Mean of each feature before scaling: {feature_means_before_scaling.round(2)}")

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print(f"Shape of feature matrix after imputation: {X.shape}")
print(f"Label shape: {y.shape}")
print(f"Concentration shape: {concentrations.shape}\n")

print("Class Distribution:")
print(tabulate(pd.Series(y).value_counts().reset_index().rename(columns={'index': 'Gas Label', 0: 'Count'}),
               headers=['Gas Label', 'Count'], tablefmt='pretty', showindex=False))
print(f"Total NaN values in X after imputation: {np.isnan(X).sum()}\n")



# Step 2.1: Feature Scaling
print("\n=== Step 2.1: Feature Scaling ===\n")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Scaling Results:")
print(f"Mean of scaled features: {X_scaled.mean(axis=0).round(5)}")
print(f"Standard deviation of scaled features: {X_scaled.std(axis=0).round(5)}\n")

# Step 2.2: Dimensionality Reduction (PCA)
print("\n=== Step 2.2: Dimensionality Reduction with PCA ===\n")
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print("PCA Results:")
print(f"Explained variance ratio for each component: {pca.explained_variance_ratio_.round(4)}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}\n")

# Step 2.3: Train-Test Split
print("\n=== Step 2.3: Train-Test Split ===\n")
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size (PCA): {X_train_pca.shape[0]}")
print(f"Test set size (PCA): {X_test_pca.shape[0]}\n")

print("Training Class Distribution:")
print(tabulate(pd.Series(y_train).value_counts().reset_index().rename(columns={'index': 'Gas Label', 0: 'Count'}),
               headers=['Gas Label', 'Count'], tablefmt='pretty', showindex=False))
print("Test Class Distribution:")
print(tabulate(pd.Series(y_test).value_counts().reset_index().rename(columns={'index': 'Gas Label', 0: 'Count'}),
               headers=['Gas Label', 'Count'], tablefmt='pretty', showindex=False))



# Step 3.1: KNN Classification
print("\n=== Step 3.1: KNN Classification ===\n")
start_time = time.time()
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
knn_training_time = time.time() - start_time
y_pred_knn = knn.predict(X_test_pca)
knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

print("KNN Model Parameters:")
print(f"Number of neighbors: {knn.n_neighbors}")
print(f"Training time: {knn_training_time:.2f} seconds")
print("KNN Performance Metrics:")
print(tabulate([['Accuracy', knn_accuracy], ['Precision', knn_precision], ['Recall', knn_recall], ['F1 Score', knn_f1]],
               headers=['Metric', 'Value'], tablefmt='pretty', floatfmt='.4f'))

# Step 3.2: Naïve Bayes Classification
print("\n=== Step 3.2: Naïve Bayes Classification ===\n")
start_time = time.time()
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
nb_training_time = time.time() - start_time
y_pred_nb = nb.predict(X_test_scaled)
nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb, average='weighted')
nb_recall = recall_score(y_test, y_pred_nb, average='weighted')
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')

print(f"Training time: {nb_training_time:.2f} seconds")
print("Naïve Bayes Performance Metrics:")
print(tabulate([['Accuracy', nb_accuracy], ['Precision', nb_precision], ['Recall', nb_recall], ['F1 Score', nb_f1]],
               headers=['Metric', 'Value'], tablefmt='pretty', floatfmt='.4f'))

# Step 3.3: Random Forest Classification
print("\n=== Step 3.3: Random Forest Classification ===\n")
start_time = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
rf_training_time = time.time() - start_time
y_pred_rf = rf.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted')
rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

rf_precision_per_class = precision_score(y_test, y_pred_rf, average=None)
rf_recall_per_class = recall_score(y_test, y_pred_rf, average=None)
gas_names = {1: 'Ethanol', 2: 'Ethylene', 3: 'Ammonia', 4: 'Acetaldehyde', 5: 'Acetone', 6: 'Toluene'}
per_class_metrics = [[gas_names[i+1], p, r] for i, (p, r) in enumerate(zip(rf_precision_per_class, rf_recall_per_class))]
feature_importance = [[f'Feature {i}', f] for i, f in enumerate(rf.feature_importances_)]

print("Random Forest Model Parameters:")
print(f"Number of trees: {rf.n_estimators}")
print(f"Training time: {rf_training_time:.2f} seconds")
print("Random Forest Performance Metrics:")
print(tabulate([['Accuracy', rf_accuracy], ['Precision', rf_precision], ['Recall', rf_recall], ['F1 Score', rf_f1]],
               headers=['Metric', 'Value'], tablefmt='pretty', floatfmt='.4f'))
print("Per-Class Precision and Recall for Random Forest:")
print(tabulate(per_class_metrics, headers=['Gas', 'Precision', 'Recall'], tablefmt='pretty', floatfmt='.4f'))
print("Feature Importance for Random Forest:")
print(tabulate(feature_importance, headers=['Feature', 'Importance'], tablefmt='pretty', floatfmt='.4f'))

# Step 3.4: K-Means Clustering
print("\n=== Step 3.4: K-Means Clustering ===\n")
start_time = time.time()
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train_pca)
kmeans_training_time = time.time() - start_time
y_pred_kmeans = kmeans.predict(X_test_pca)
kmeans_ari = adjusted_rand_score(y_test, y_pred_kmeans)
kmeans_silhouette = silhouette_score(X_test_pca, y_pred_kmeans)

print("K-Means Model Parameters:")
print(f"Number of clusters: {kmeans.n_clusters}")
print(f"Training time: {kmeans_training_time:.2f} seconds")
print("K-Means Performance Metrics:")
print(tabulate([['Adjusted Rand Index', kmeans_ari], ['Silhouette Score', kmeans_silhouette]],
               headers=['Metric', 'Value'], tablefmt='pretty', floatfmt='.4f'))

# Confusion Matrix Table
print("\nConfusion Matrix for Random Forest (Table):")
cm = confusion_matrix(y_test, y_pred_rf)
cm_table = [[gas_names.get(i+1, i+1), gas_names.get(j+1, j+1), cm[i, j]] for i in range(6) for j in range(6)]
print(tabulate(cm_table, headers=['True Label', 'Predicted Label', 'Count'], tablefmt='pretty'))



# Step 4: Visualization and Analysis
print("\n=== Step 4: Visualization and Analysis ===\n")

# 4.1 Class Distribution Plot with Custom Colors
print("Generating Class Distribution Plot...")
class_counts = pd.Series(y).value_counts()
gas_colors = {'Ethanol': 'blue', 'Ethylene': 'green', 'Ammonia': 'red', 'Acetaldehyde': 'purple', 'Acetone': 'orange', 'Toluene': 'brown'}
plt.figure(figsize=(8, 6))
bars = plt.bar(class_counts.index.map(gas_names), class_counts.values, color=[gas_colors[gas_names[i]] for i in class_counts.index])
plt.xlabel('Gas Class')
plt.ylabel('Number of Instances')
plt.title('Class Distribution of Gas Data')
plt.xticks(rotation=45)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{int(height)}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('class_distribution.png')
plt.show()

# 4.2 PCA Scatter Plot
print("Generating PCA Scatter Plot...")
plt.figure(figsize=(10, 6))
gas_colors = {1: 'blue', 2: 'green', 3: 'red', 4: 'purple', 5: 'orange', 6: 'brown'}
for gas_label in range(1, 7):
    mask = (y == gas_label)
    plt.scatter(X_pca[mask, 0], X_pca[mask, 1], label=gas_names[gas_label], alpha=0.6, color=gas_colors[gas_label])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA Scatter Plot of Gas Data')
plt.legend()
plt.grid(True)
plt.savefig('pca_scatter_plot.png')
plt.show()

# 4.3 Confusion Matrix for Random Forest
print("Generating Confusion Matrix for Random Forest...")
cm = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=list(gas_names.values()), yticklabels=list(gas_names.values()))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for Random Forest')
plt.savefig('confusion_matrix_rf.png')
plt.show()

# 4.4 Bar Chart of Algorithm Performance
print("Generating Comparison of Classification Algorithms...")
algorithms = ['KNN', 'Naïve Bayes', 'Random Forest']
accuracies = [knn_accuracy, nb_accuracy, rf_accuracy]
plt.figure(figsize=(8, 6))
plt.bar(algorithms, accuracies, color=['skyblue', 'lightgreen', 'salmon'])
plt.xlabel('Algorithm')
plt.ylabel('Accuracy')
plt.title('Comparison of Classification Algorithms')
plt.ylim(0, 1)
for i, v in enumerate(accuracies):
    plt.text(i, v + 0.02, f'{v:.4f}', ha='center')
plt.savefig('algorithm_performance.png')
plt.show()

# 4.5 Feature Importance Plot for Random Forest
print("Generating Feature Importance Plot for Random Forest...")
plt.figure(figsize=(10, 6))
feature_importances = rf.feature_importances_
indices = np.argsort(feature_importances)[::-1]  # Sort features by importance
top_n = 20  # Plot top 20 features for readability
plt.bar(range(top_n), feature_importances[indices[:top_n]], color='salmon')
plt.xticks(range(top_n), [f'Feature {i}' for i in indices[:top_n]], rotation=45, ha='right')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Top 20 Feature Importances for Random Forest')
plt.tight_layout()
plt.savefig('feature_importance_rf.png')
plt.show()

# 4.6 Training Time Plot for Algorithms
print("Generating Training Time Plot for Algorithms...")
training_times = [knn_training_time, nb_training_time, rf_training_time, kmeans_training_time]
algorithms_with_kmeans = ['KNN', 'Naïve Bayes', 'Random Forest', 'K-Means']
plt.figure(figsize=(8, 6))
plt.bar(algorithms_with_kmeans, training_times, color=['skyblue', 'lightgreen', 'salmon', 'gray'])
plt.xlabel('Algorithm')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time Comparison of Algorithms')
for i, v in enumerate(training_times):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.savefig('training_time_comparison.png')
plt.show()