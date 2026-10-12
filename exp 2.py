"""
Experiment 2: Covariance, Eigenvalues and Eigenvectors Analysis
----------------------------------------------------------------
Aim:
    Compute a covariance matrix and its eigenvalues and eigenvectors,
    then use them for principal component analysis (PCA), a standard
    way to reduce the number of features before a neural network.

Operations covered:
    Mean centering, covariance, eigenvalues, eigenvectors
    Eigen-equation check  A v = λ v
    PCA projection and explained variance
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Sample data
# ---------------------------------------------------------------------------
section("1. Sample data")

# Four samples, two features. Feature 2 is roughly twice feature 1,
# so the data lies along one main direction.
X = np.array([[1.0, 2.1],
              [2.0, 3.9],
              [3.0, 6.2],
              [4.0, 7.8]])

print("Data X (samples, features):\n", X)
print("Shape:", X.shape)


# ---------------------------------------------------------------------------
# 2. Mean and centering
# ---------------------------------------------------------------------------
section("2. Mean and centering")

mean = np.mean(X, axis=0)
X_centered = X - mean

print("Feature means          :", mean)
print("Centered data:\n", X_centered)


# ---------------------------------------------------------------------------
# 3. Covariance matrix
# ---------------------------------------------------------------------------
section("3. Covariance matrix")

# rowvar=False: each column is a feature, each row is a sample
cov = np.cov(X_centered, rowvar=False)

print("Covariance matrix:\n", cov)
print("Variance of feature 1  :", cov[0, 0])
print("Variance of feature 2  :", cov[1, 1])
print("Covariance of 1 and 2  :", cov[0, 1])


# ---------------------------------------------------------------------------
# 4. Eigenvalues and eigenvectors
# ---------------------------------------------------------------------------
section("4. Eigenvalues and eigenvectors")

# eigh is for symmetric matrices such as a covariance matrix,
# and it returns real eigenvalues.
eigenvalues, eigenvectors = np.linalg.eigh(cov)

# Sort from largest eigenvalue to smallest so the first column
# is the direction of greatest variance (the first principal component).
order = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[order]
eigenvectors = eigenvectors[:, order]

print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors (columns):\n", eigenvectors)

# The eigen-equation: covariance @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    left = cov @ v
    right = lam * v
    print(f"\nComponent {i + 1}")
    print("  cov @ v :", left)
    print("  λ * v   :", right)
    print("  match   :", np.allclose(left, right))


# ---------------------------------------------------------------------------
# 5. Explained variance
# ---------------------------------------------------------------------------
section("5. Explained variance")

total = np.sum(eigenvalues)
ratios = eigenvalues / total

print("Total variance              :", total)
for i, ratio in enumerate(ratios):
    print(f"Component {i + 1} explains       : {ratio * 100:.2f}%")
print("Cumulative (first component):", ratios[0] * 100, "%")


# ---------------------------------------------------------------------------
# 6. PCA projection
# ---------------------------------------------------------------------------
section("6. PCA projection")

# Keep only the first principal component
k = 1
components = eigenvectors[:, :k]

# Project centered data onto that direction: (n, d) @ (d, k) -> (n, k)
Z = X_centered @ components

# Reconstruct the original scale by mapping back and adding the mean
X_hat = Z @ components.T + mean

print("Principal component:\n", components)
print("Projected scores Z:\n", Z)
print("Reconstructed X:\n", X_hat)
print("Reconstruction error (MSE):", np.mean((X - X_hat) ** 2))

print("\nExperiment 2 complete.")
