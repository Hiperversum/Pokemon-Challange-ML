"""Logistic regression trained with our own gradient descent (Optimization slides).

The model minimizes the negative log-likelihood (NLL) of the logistic model,
plus an optional L2 penalty, updating the weights against the gradient:

    beta <- beta - learning_rate * gradient

With batch_size=None every step uses the whole training set (batch GD),
otherwise each step uses a random mini-batch (mini-batch GD).
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    """Map any real number to a probability between 0 and 1."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class GradientDescentLogisticRegression(ClassifierMixin, BaseEstimator):
    """Binary logistic regression fitted with batch or mini-batch gradient descent."""

    def __init__(self, learning_rate=0.1, n_epochs=300, batch_size=None, l2=0.0, random_state=0):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2 = l2
        self.random_state = random_state

    def _add_bias(self, X):
        """Add a column of ones so the first weight works as the intercept."""
        X = np.asarray(X, dtype=float)
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _cost(self, Xb, y, beta):
        """Mean negative log-likelihood plus the L2 penalty (the bias is not penalized)."""
        p = np.clip(sigmoid(Xb @ beta), 1e-12, 1 - 1e-12)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        return nll + 0.5 * self.l2 * np.sum(beta[1:] ** 2)

    def _gradient(self, Xb, y, beta):
        """Gradient of the cost: X^T (sigmoid(X beta) - y) / n, plus l2 * beta."""
        grad = Xb.T @ (sigmoid(Xb @ beta) - y) / len(y)
        grad[1:] += self.l2 * beta[1:]
        return grad

    def fit(self, X, y):
        """Run gradient descent and store the weights and the cost of every epoch."""
        Xb = self._add_bias(X)
        y = np.asarray(y, dtype=float)
        self.classes_ = np.array([0, 1])
        rng = np.random.default_rng(self.random_state)
        beta = np.zeros(Xb.shape[1])
        batch = self.batch_size or len(y)
        self.cost_history_ = [self._cost(Xb, y, beta)]

        for _ in range(self.n_epochs):
            order = rng.permutation(len(y)) if self.batch_size else np.arange(len(y))
            for start in range(0, len(y), batch):
                idx = order[start:start + batch]
                beta -= self.learning_rate * self._gradient(Xb[idx], y[idx], beta)
            self.cost_history_.append(self._cost(Xb, y, beta))

        self.intercept_ = beta[0]
        self.coef_ = beta[1:]
        return self

    def predict_proba(self, X):
        """Return the probabilities of class 0 and class 1."""
        p = sigmoid(self._add_bias(X) @ np.concatenate([[self.intercept_], self.coef_]))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """Predict 1 when the probability of a Player 1 win is at least 0.5."""
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
