"""
Linear Regression from Scratch
---------------------------------
Domain: AI/ML
OOP: Model, Layer (LinearLayer), LossFunction
DSA/Concepts: gradient descent, vector/matrix operations (pure python, no numpy)
"""

import random


class LossFunction:
    """Mean Squared Error loss and its gradient."""

    @staticmethod
    def compute(y_true: list, y_pred: list) -> float:
        n = len(y_true)
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

    @staticmethod
    def gradient(y_true: list, y_pred: list) -> list:
        """dLoss/dy_pred for each prediction."""
        n = len(y_true)
        return [(2 / n) * (yp - yt) for yt, yp in zip(y_true, y_pred)]


class Layer:
    """A single linear layer: y = w . x + b (supports multiple features)."""

    def __init__(self, num_features: int):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(num_features)]
        self.bias = 0.0

    def forward(self, x: list) -> float:
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias

    def forward_batch(self, X: list) -> list:
        return [self.forward(x) for x in X]

    def update(self, dw: list, db: float, learning_rate: float):
        self.weights = [w - learning_rate * d for w, d in zip(self.weights, dw)]
        self.bias -= learning_rate * db


class Model:
    """Linear Regression model trained via batch gradient descent."""

    def __init__(self, num_features: int, learning_rate: float = 0.01, epochs: int = 1000):
        self.layer = Layer(num_features)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_history = []

    def _compute_gradients(self, X: list, y_true: list, y_pred: list):
        grad_output = LossFunction.gradient(y_true, y_pred)
        n_features = len(X[0])
        n_samples = len(X)

        dw = [0.0] * n_features
        db = 0.0

        for i in range(n_samples):
            for j in range(n_features):
                dw[j] += grad_output[i] * X[i][j]
            db += grad_output[i]

        dw = [d / n_samples for d in dw]
        db /= n_samples
        return dw, db

    def fit(self, X: list, y: list, verbose: bool = False):
        for epoch in range(self.epochs):
            y_pred = self.layer.forward_batch(X)
            loss = LossFunction.compute(y, y_pred)
            self.loss_history.append(loss)

            dw, db = self._compute_gradients(X, y, y_pred)
            self.layer.update(dw, db, self.learning_rate)

            if verbose and epoch % (self.epochs // 10 or 1) == 0:
                print(f"Epoch {epoch:4d} | Loss: {loss:.6f}")

    def predict(self, X: list) -> list:
        return self.layer.forward_batch(X)

    def score(self, X: list, y: list) -> float:
        """R-squared score."""
        y_pred = self.predict(X)
        mean_y = sum(y) / len(y)
        ss_total = sum((yi - mean_y) ** 2 for yi in y)
        ss_residual = sum((yi - ypi) ** 2 for yi, ypi in zip(y, y_pred))
        return 1 - (ss_residual / ss_total) if ss_total else 0.0


def normalize(X: list):
    """Min-max normalize features (returns normalized X and the scalers)."""
    n_features = len(X[0])
    mins = [min(row[j] for row in X) for j in range(n_features)]
    maxs = [max(row[j] for row in X) for j in range(n_features)]

    norm_X = []
    for row in X:
        norm_row = [
            (row[j] - mins[j]) / (maxs[j] - mins[j]) if maxs[j] != mins[j] else 0
            for j in range(n_features)
        ]
        norm_X.append(norm_row)
    return norm_X, mins, maxs


def main():
    random.seed(1)
    # Synthetic data: y = 3x + 5 with noise
    X_raw = [[x] for x in range(1, 21)]
    y = [3 * x[0] + 5 + random.uniform(-1, 1) for x in X_raw]

    X, mins, maxs = normalize(X_raw)

    model = Model(num_features=1, learning_rate=0.5, epochs=500)

    print("=== Linear Regression from Scratch Demo ===")
    model.fit(X, y, verbose=True)

    r2 = model.score(X, y)
    print(f"\nFinal weights: {model.layer.weights}")
    print(f"Final bias: {model.layer.bias:.4f}")
    print(f"R^2 score: {r2:.4f}")

    test_x_raw = 25
    test_x_norm = (test_x_raw - mins[0]) / (maxs[0] - mins[0])
    prediction = model.predict([[test_x_norm]])[0]
    print(f"\nPrediction for x={test_x_raw}: y = {prediction:.2f} (expected ~{3*test_x_raw+5})")


if __name__ == "__main__":
    main()
