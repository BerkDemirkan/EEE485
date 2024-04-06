# [1]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, prange


# [2]
data = pd.read_csv("processed_database_2.csv", low_memory=False)
data.reset_index(drop=True, inplace=True)

# data = data.sample(n=100000)

# Train/Test Split
data_train = data.sample(frac=0.8)
data_test = data.drop(data_train.index)

X_train = data_train.iloc[:, 1:].to_numpy()
X_test = data_test.iloc[:, 1:].to_numpy()

Y_train = data_train.iloc[:, :1].values
Y_test = data_test.iloc[:, :1].values

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)


# [3]
@jit(nopython=True)
def mse(Y_true: np.ndarray, Y_pred: np.ndarray):
    return np.square(Y_true - Y_pred).mean()


# [4]
@jit(nopython=True)
def find_best_split(X: np.ndarray, Y: np.ndarray):
    best_feature, best_threshold, best_error = None, None, np.inf
    for feature_idx in prange(X.shape[1]):
        feature_values = X[:, feature_idx]
        feature_values.sort()
        for threshold in feature_values:
            Y_left = Y[X[:, feature_idx] <= threshold]
            Y_right = Y[X[:, feature_idx] > threshold]

            if len(Y_left) == 0 or len(Y_right) == 0:
                continue

            error_left = mse(Y_left, np.mean(Y_left))
            error_right = mse(Y_right, np.mean(Y_right))
            weighted_error = len(Y_left) * error_left + len(Y_right) * error_right

            if weighted_error < best_error:
                best_error = weighted_error
                best_feature = feature_idx
                best_threshold = threshold
                break  # Break out of the loop once the best error is found

    return best_feature, best_threshold


# [5]
class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self._depth = 0

    def _build_three(self, X: np.ndarray, Y: np.ndarray, depth=0, verbose=False):
        self._depth = max(self._depth, depth)

        if verbose:
            print(f"Current Depth: {depth}, Max Depth: {self._depth}        ", end="\r")

        if depth >= self.max_depth or len(np.unique(Y)) == 1:
            return np.mean(Y)

        feature, threshold = find_best_split(X, Y)
        if feature is None:
            return np.mean(Y)

        left_mask = X[:, feature] <= threshold
        right_mask = ~left_mask
        left_child = self._build_three(
            X[left_mask], Y[left_mask], depth + 1, verbose=verbose
        )
        right_child = self._build_three(
            X[right_mask], Y[right_mask], depth + 1, verbose=verbose
        )

        return {
            "feature": feature,
            "threshold": threshold,
            "left": left_child,
            "right": right_child,
        }

    def fit(self, X, Y, verbose=False):
        self._depth = 0
        self.tree_ = self._build_three(X, Y, verbose=verbose)

    def _predict_single(self, sample, node):
        if not isinstance(node, dict):
            return node

        feature, threshold = node["feature"], node["threshold"]

        if sample[feature] <= threshold:
            return self._predict_single(sample, node["left"])
        else:
            return self._predict_single(sample, node["right"])

    def predict(self, X):
        pred = np.array([self._predict_single(sample, self.tree_) for sample in X])
        return pred.reshape(pred.shape[0], 1)


# [6]
regressor = DecisionTree(max_depth=np.inf)

rand_idxs = np.random.choice(X_train.shape, (1000))
regressor.fit(X_train, Y_train, verbose=True)


# [7]
Y_pred_train = regressor.predict(X_train)
residuals_train = Y_train.T - Y_pred_train
train_cost = mse(Y_train.T, Y_pred_train)
print(f"Training cost: {np.squeeze(train_cost).round(2)}")

fig, ax = plt.subplots()
ax.scatter(Y_train, Y_pred_train, (0.01))
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Actual vs Prediction (Decision Tree, Train)")

ax.plot(np.arange(100), np.arange(100), "r")
fig.savefig("tree_train_pred")


fig, ax = plt.subplots()
ax.set_autoscale_on(False)
ax.scatter(Y_train, residuals_train, (0.01))
ax.set_xbound(-50, 100)
ax.set_ybound(-100, 100)
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")
ax.set_title("Predicted Value vs Residual (Decision Tree, Train)")

ax.plot(np.arange(150) - 50, np.zeros((150,)), "black")

fig.savefig("tree_train_resd")


# [8]
Y_pred_test = regressor.predict(X_train)
residuals_test = Y_test.T - Y_pred_test
test_cost = mse(Y_test.T, Y_pred_test)
print(f"Training cost: {np.squeeze(test_cost).round(2)}")

fig, ax = plt.subplots()
ax.scatter(Y_test, Y_pred_test, (0.01))
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Actual vs Prediction (Decision Tree, Test)")

ax.plot(np.arange(100), np.arange(100), "r")
fig.savefig("tree_test_pred")


fig, ax = plt.subplots()
ax.set_autoscale_on(False)
ax.scatter(Y_pred_test, residuals_test, (0.01))
ax.set_xbound(-50, 100)
ax.set_ybound(-100, 100)
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")
ax.set_title("Predicted Value vs Residual (Decision Tree, Test)")

ax.plot(np.arange(150) - 50, np.zeros((150,)), "black")

fig.savefig("tree_test_resd")


# [9]
class RandomForest:
    def __init__(self, n_trees, max_depth):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self._trees = []

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose=False):
        self.trees = []
        for i in range(self.n_trees):
            if verbose:
                print(f"\nTree: {i+1}/{self.n_trees}")

            sample_idxs = np.random.choice(
                np.arange(X.shape[0]), int(X.shape[0] / self.n_trees), replace=True
            )

            tree = DecisionTree(max_depth=self.max_depth)
            X_sample = X[sample_idxs]
            Y_sample = Y[sample_idxs]
            tree.fit(X_sample, Y_sample, verbose=verbose)
            self._trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self._trees])
        return tree_preds.mean(axis=0)


# [10]
forest = RandomForest(n_trees=100, max_depth=np.inf)
forest.fit(X_train, Y_train, verbose=True)
