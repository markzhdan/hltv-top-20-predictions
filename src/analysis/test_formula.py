"""
CS 412 Final Project – Predicting HLTV Top-20 CS2 Player Rankings
Author: Mark Zhdan (mzhda3)
Updated: 2025-05-09  —  bug-fixed loader, stable training, prints 2024 predictions

Run:
```bash
python cs412_project.py "data/clean/complex/final/player_data_*.csv"
```
Only **NumPy** and **Pandas** required.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import re, glob, os, sys
from typing import Tuple, List, Optional, Dict

# ---------------------------------------------------------------------------
# Column names (edit if schema differs)
# ---------------------------------------------------------------------------
LABEL_COL = "Rank"  # 1, 2, … – will shift to 0-based
PLAYER_COL = "Player"
YEAR_COL = "Year"  # injected from filename if absent

NUM_CLASSES: int = 20  # set dynamically after loading labels

# ---------------------------------------------------------------------------
# Robust CSV loading helpers
# ---------------------------------------------------------------------------


def _read_single_csv(path: str) -> pd.DataFrame:
    """Read CSV with bad-line skipping & infer Year from filename."""
    try:
        df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    except TypeError:
        df = pd.read_csv(
            path, engine="python", error_bad_lines=False, warn_bad_lines=True
        )
    if YEAR_COL not in df.columns:
        m = re.search(r"(20\d{2})", os.path.basename(path))
        if m:
            df[YEAR_COL] = int(m.group(1))
    return df


def load_dataset(
    pattern: str, feature_drop: Optional[List[str]] = None
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load & concatenate all CSVs matching *pattern*. Impute NaNs & infs."""
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No files match pattern {pattern}")
    df = pd.concat([_read_single_csv(p) for p in paths], ignore_index=True)

    # build drop list for features
    if feature_drop is None:
        feature_drop = []
    for col in (LABEL_COL, PLAYER_COL):
        if col in df.columns:
            feature_drop.append(col)

    # numeric features
    numeric = df.select_dtypes(include=[np.number])
    numeric = numeric.replace([np.inf, -np.inf], np.nan)
    numeric = numeric.fillna(numeric.mean())

    X = numeric.drop(
        columns=[c for c in feature_drop if c in numeric.columns], errors="ignore"
    ).values.astype(float)

    if LABEL_COL not in df.columns:
        raise ValueError(f"Missing column {LABEL_COL}")
    y = df[LABEL_COL].astype(int).values - 1

    global NUM_CLASSES
    NUM_CLASSES = int(y.max()) + 1
    return X, y, df


# ---------------------------------------------------------------------------
# Pre-processing helpers
# ---------------------------------------------------------------------------


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-8
    return (X - mu) / sigma, mu, sigma


def one_hot(y: np.ndarray, k: int) -> np.ndarray:
    out = np.zeros((y.size, k))
    out[np.arange(y.size), y] = 1
    return out


# ---------------------------------------------------------------------------
# Models – Softmax Regression & Simple NN
# ---------------------------------------------------------------------------


class SoftmaxRegression:
    def __init__(self, input_dim: int, num_classes: int, lr=0.01, l2=1e-3):
        self.W = np.zeros((input_dim, num_classes))
        self.b = np.zeros((1, num_classes))
        self.lr, self.l2 = lr, l2

    @staticmethod
    def _softmax(z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, X):
        return self._softmax(X @ self.W + self.b)

    def loss(self, X, y):
        m = X.shape[0]
        p = self.forward(X)
        eps = 1e-9
        return -np.log(p[np.arange(m), y] + eps).mean() + 0.5 * self.l2 * np.sum(
            self.W * self.W
        )

    def fit(self, X, y, epochs=300, batch=32, verbose=True):
        m = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(m)
            for start in range(0, m, batch):
                b = idx[start : start + batch]
                xb, yb = X[b], y[b]
                p = self.forward(xb)
                if np.isnan(p).any():
                    continue  # skip unstable batch
                oh = one_hot(yb, self.W.shape[1])
                dW = xb.T @ (p - oh) / xb.shape[0] + self.l2 * self.W
                db = (p - oh).mean(axis=0, keepdims=True)
                self.W -= self.lr * dW
                self.b -= self.lr * db
            if verbose and ((ep + 1) % 30 == 0 or ep == 0):
                print(f"[Softmax] {ep+1:03}/{epochs}  loss={self.loss(X,y):.4f}")

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


class SimpleNN:
    def __init__(self, input_dim: int, hidden: int, num_classes: int, lr=0.01, l2=1e-4):
        self.W1 = np.random.randn(input_dim, hidden) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, num_classes) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros((1, num_classes))
        self.lr, self.l2 = lr, l2

    @staticmethod
    def _relu(z):
        return np.maximum(0, z)

    @staticmethod
    def _relu_deriv(z):
        return (z > 0).astype(float)

    @staticmethod
    def _softmax(z):
        exp = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp / exp.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self._relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self._softmax(self.z2)
        return self.a2

    def loss(self, X, y):
        m = X.shape[0]
        p = self.forward(X)
        eps = 1e-9
        data = -np.log(p[np.arange(m), y] + eps).mean()
        reg = 0.5 * self.l2 * (np.sum(self.W1 * self.W1) + np.sum(self.W2 * self.W2))
        return data + reg

    def fit(self, X, y, epochs=400, batch=32, verbose=True):
        m = X.shape[0]
        for ep in range(epochs):
            idx = np.random.permutation(m)
            for start in range(0, m, batch):
                b = idx[start : start + batch]
                xb, yb = X[b], y[b]
                p = self.forward(xb)
                oh = one_hot(yb, self.W2.shape[1])
                dz2 = (p - oh) / xb.shape[0]
                if np.isnan(dz2).any():
                    continue
                dW2 = self.a1[b - start].T @ dz2 + self.l2 * self.W2
                db2 = dz2.sum(axis=0, keepdims=True)
                da1 = dz2 @ self.W2.T
                dz1 = da1 * self._relu_deriv(self.z1[b - start])
                dW1 = xb.T @ dz1 + self.l2 * self.W1
                db1 = dz1.sum(axis=0, keepdims=True)
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
            if verbose and ((ep + 1) % 40 == 0 or ep == 0):
                print(f"[NN] {ep+1:03}/{epochs}  loss={self.loss(X,y):.4f}")

    def predict(self, X):
        return self.forward(X).argmax(axis=1)


# ---------------------------------------------------------------------------
# Main driver – prints 2024 predictions
# ---------------------------------------------------------------------------
SPLIT_BY_YEAR = True
TEST_YEAR = 2024
DEFAULT_PATTERN = "data/clean/complex/final/player_data_*.csv"

if __name__ == "__main__":
    pattern = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATTERN
    print("Loading pattern:", pattern)
    X_raw, y_raw, df = load_dataset(pattern)
    X, _, _ = standardize(X_raw)

    if SPLIT_BY_YEAR and YEAR_COL in df.columns:
        train_mask = df[YEAR_COL] < TEST_YEAR
        test_mask = df[YEAR_COL] == TEST_YEAR
    else:
        m = len(X)
        idx = np.random.permutation(m)
        split = int(0.8 * m)
        train_mask = np.zeros(m, dtype=bool)
        train_mask[idx[:split]] = True
        test_mask = ~train_mask

    X_train, y_train = X[train_mask], y_raw[train_mask]
    X_test, y_test = X[test_mask], y_raw[test_mask]

    # ---------------- Softmax ----------------
    soft = SoftmaxRegression(X.shape[1], NUM_CLASSES, lr=0.01, l2=1e-3)
    soft.fit(X_train, y_train, epochs=300)
    print("Softmax train acc:", (soft.predict(X_train) == y_train).mean())
    print("Softmax test  acc:", (soft.predict(X_test) == y_test).mean())

    if YEAR_COL in df.columns:
        df_pred = df[test_mask][[PLAYER_COL, YEAR_COL]].copy()
        df_pred["Pred"] = soft.predict(X_test) + 1
        df_pred = df_pred.sort_values("Pred").reset_index(drop=True)
        print("Predicted 2024 Player Rankings (Softmax):")
        print(df_pred[[PLAYER_COL, "Pred"]].to_string(index=False))
