import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Step 1: Load and Combine the Data
# Load all 10 batches
batches = []
for i in range(1, 11):
    batch = pd.read_csv(f'batch{i}.dat', sep=' ', header=None)
    batches.append(batch)

# Combine into a single DataFrame
data = pd.concat(batches, ignore_index=True)

# Extract gas labels from the first column (split on ';')
data[0] = data[0].astype(str)  # Ensure the column is treated as a string
y = data[0].apply(lambda x: int(x.split(';')[0]))  # Extract the gas label

# Extract concentrations from the first column (after the ';')
concentrations = data[0].apply(lambda x: float(x.split(';')[1]))

# Extract the secondary concentration from the second column (after '1:')
concentrations_secondary = data[1].apply(lambda x: float(x.split(':')[1]))

# Extract features (columns 2 to 129: 128 features)
feature_columns = data.iloc[:, 2:130]  # Columns 2 to 129 (128 features)
X = np.zeros((data.shape[0], 128))  # Initialize X with the correct shape (13910, 128)
for col_idx in range(128):
    column = feature_columns.iloc[:, col_idx].astype(str)
    X[:, col_idx] = column.apply(lambda x: float(x.split(':')[1]) if ':' in x else float(x) if x != 'nan' else np.nan)

# Impute NaN values with the mean of each column
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Verify shapes and check for NaN
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