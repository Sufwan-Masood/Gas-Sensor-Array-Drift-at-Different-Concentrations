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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, adjusted_rand_score

# Step 1: Load and Combine the Data
batches = []
for i in range(1, 11):
    batch = pd.read_csv(f'batch{i}.dat', sep=' ', header=None)
    batches.append(batch)

data = pd.concat(batches, ignore_index=True)

data[0] = data[0].astype(str)
y = data[0].apply(lambda x: int(x.split(';')[0]))
concentrations = data[0].apply(lambda x: float(x.split(';')[1]))
concentrations_secondary = data[1].apply(lambda x: float(x.split(':')[1]))

feature_columns = data.iloc[:, 2:130]
X = np.zeros((data.shape[0], 128))
for col_idx in range(128):
    column = feature_columns.iloc[:, col_idx].astype(str)
    X[:, col_idx] = column.apply(lambda x: float(x.split(':')[1]) if ':' in x else float(x) if x != 'nan' else np.nan)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Concentrations shape: {concentrations.shape}")
print("Class distribution:\n", pd.Series(y).value_counts())
print("First few rows of X:\n", X[:5, :5])
print(f"Total NaN values in X after imputation: {np.isnan(X).sum()}")

# Step 2.1: Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("Mean of scaled features:", X_scaled.mean(axis=0).round(5))
print("Std of scaled features:", X_scaled.std(axis=0).round(5))

# Step 2.2: Dimensionality Reduction (PCA)
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)
print(f"Explained variance ratio (first 10 components): {pca.explained_variance_ratio_}")
print(f"Total explained variance: {pca.explained_variance_ratio_.sum():.4f}")

# Step 2.3: Train-Test Split
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
X_train_scaled, X_test_scaled, _, _ = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
print(f"Training set size (PCA): {X_train_pca.shape[0]}")
print(f"Test set size (PCA): {X_test_pca.shape[0]}")
print("Training class distribution:\n", pd.Series(y_train).value_counts())
print("Test class distribution:\n", pd.Series(y_test).value_counts())

# Step 3.1: KNN Classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_pca, y_train)
y_pred_knn = knn.predict(X_test_pca)

knn_accuracy = accuracy_score(y_test, y_pred_knn)
knn_precision = precision_score(y_test, y_pred_knn, average='weighted')
knn_recall = recall_score(y_test, y_pred_knn, average='weighted')
knn_f1 = f1_score(y_test, y_pred_knn, average='weighted')

print("KNN Results:")
print(f"Accuracy: {knn_accuracy:.4f}")
print(f"Precision: {knn_precision:.4f}")
print(f"Recall: {knn_recall:.4f}")
print(f"F1 Score: {knn_f1:.4f}")

# Step 3.2: Naïve Bayes Classification
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
y_pred_nb = nb.predict(X_test_scaled)

nb_accuracy = accuracy_score(y_test, y_pred_nb)
nb_precision = precision_score(y_test, y_pred_nb, average='weighted')
nb_recall = recall_score(y_test, y_pred_nb, average='weighted')
nb_f1 = f1_score(y_test, y_pred_nb, average='weighted')

print("Naïve Bayes Results:")
print(f"Accuracy: {nb_accuracy:.4f}")
print(f"Precision: {nb_precision:.4f}")
print(f"Recall: {nb_recall:.4f}")
print(f"F1 Score: {nb_f1:.4f}")

# Step 3.3: Random Forest Classification
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_precision = precision_score(y_test, y_pred_rf, average='weighted')
rf_recall = recall_score(y_test, y_pred_rf, average='weighted')
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

print("Random Forest Results:")
print(f"Accuracy: {rf_accuracy:.4f}")
print(f"Precision: {rf_precision:.4f}")
print(f"Recall: {rf_recall:.4f}")
print(f"F1 Score: {rf_f1:.4f}")

# Step 3.4: K-Means Clustering
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(X_train_pca)
y_pred_kmeans = kmeans.predict(X_test_pca)

kmeans_ari = adjusted_rand_score(y_test, y_pred_kmeans)

print("K-Means Results:")
print(f"Adjusted Rand Index: {kmeans_ari:.4f}")