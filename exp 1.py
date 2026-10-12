"""
Experiment 1: Vector and Matrix Operations for Deep Learning
-----------------------------------------------------------
Aim:
    Implement the vector and matrix operations that form the basis of
    neural-network computation, using NumPy.

Operations covered:
    Vectors  - creation, addition, subtraction, scalar multiplication,
               dot product, L2 norm, cosine similarity
    Matrices - creation, addition, multiplication, transpose,
               element-wise (Hadamard) product, determinant, inverse
    Deep learning link - a single linear layer: y = Wx + b
"""

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def section(title):
    print(f"\n{'=' * 60}")
    print(title)
    print("=" * 60)


# ---------------------------------------------------------------------------
# 1. Vector operations
# ---------------------------------------------------------------------------
section("1. Vector operations")

# A feature vector x and a weight vector w, as used in a neuron
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.2, -0.5, 0.8])

print("Feature vector x :", x)
print("Weight vector  w :", w)

print("\nAddition          x + w :", x + w)
print("Subtraction       x - w :", x - w)
print("Scalar multiply   2 * x :", 2 * x)

# Dot product: weighted sum of inputs, the core of a neuron
dot = np.dot(x, w)
print("\nDot product       x · w :", dot)
print("Manual check            :", np.sum(x * w))

# L2 (Euclidean) norm: magnitude of a vector
l2 = np.linalg.norm(x)
print("\nL2 norm of x            :", l2)

# Cosine similarity: how aligned two vectors are (used in embeddings)
cosine = np.dot(x, w) / (np.linalg.norm(x) * np.linalg.norm(w))
print("Cosine similarity x, w  :", cosine)


# ---------------------------------------------------------------------------
# 2. Matrix operations
# ---------------------------------------------------------------------------
section("2. Matrix operations")

# Weight matrix W maps 3 input features to 2 neurons
A = np.array([[1.0, 2.0, 3.0],
              [4.0, 5.0, 6.0]])
B = np.array([[0.1, 0.2, 0.3],
              [0.4, 0.5, 0.6]])

print("Matrix A:\n", A)
print("Matrix B:\n", B)

print("\nAddition A + B:\n", A + B)
print("Subtraction A - B:\n", A - B)

# Transpose: rows become columns (needed to align shapes for multiplication)
print("Transpose of A:\n", A.T)
print("Shape of A:", A.shape, "  Shape of A.T:", A.T.shape)

# Matrix multiplication: combine transformations
# A is (2, 3), A.T is (3, 2), so A @ A.T is (2, 2)
product = A @ A.T
print("\nMatrix product A @ A.T:\n", product)

# Hadamard (element-wise) product: used in gating, e.g. LSTM / attention masks
print("Element-wise product A * B:\n", A * B)

# Square matrix for determinant and inverse
C = np.array([[4.0, 7.0],
              [2.0, 6.0]])
print("\nSquare matrix C:\n", C)
print("Determinant of C :", np.linalg.det(C))
print("Inverse of C:\n", np.linalg.inv(C))
print("C @ inv(C) (identity check):\n", C @ np.linalg.inv(C))


# ---------------------------------------------------------------------------
# 3. Broadcasting
# ---------------------------------------------------------------------------
section("3. Broadcasting")

# A bias vector of shape (2,) added to a batch of activations of shape (3, 2)
activations = np.array([[1.0, 2.0],
                        [3.0, 4.0],
                        [5.0, 6.0]])
bias = np.array([0.5, -0.5])

print("Activations (batch, features):\n", activations)
print("Bias:", bias)
print("Activations + bias:\n", activations + bias)


# ---------------------------------------------------------------------------
# 4. A linear layer: y = Wx + b
# ---------------------------------------------------------------------------
section("4. Linear layer  y = W x + b")

# One sample, 3 features
x = np.array([1.0, 0.5, -1.0])

# Two neurons, each with 3 weights
W = np.array([[0.2, -0.4, 0.1],
              [0.5, 0.3, -0.2]])
b = np.array([0.1, -0.1])

# W is (2, 3), x is (3,), so W @ x is (2,)
y = W @ x + b

print("Input x          :", x)
print("Weights W:\n", W)
print("Bias b           :", b)
print("Output y = Wx + b:", y)

# Same computation for a mini-batch of 4 samples, shape (4, 3)
batch = np.array([[1.0, 0.5, -1.0],
                  [0.0, 1.0, 0.5],
                  [-0.5, 0.2, 0.8],
                  [1.5, -0.3, 0.0]])

# batch @ W.T  -> (4, 3) @ (3, 2) = (4, 2), then add bias
batch_out = batch @ W.T + b
print("\nBatch input shape :", batch.shape)
print("Batch output shape:", batch_out.shape)
print("Batch output y:\n", batch_out)


# ---------------------------------------------------------------------------
# 5. ReLU on the linear output
# ---------------------------------------------------------------------------
section("5. ReLU activation")


def relu(z):
    """Rectified Linear Unit: max(0, z)."""
    return np.maximum(0, z)


activated = relu(batch_out)
print("Before ReLU:\n", batch_out)
print("After ReLU:\n", activated)

print("\nExperiment 1 complete.")
