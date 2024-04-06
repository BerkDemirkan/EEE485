# [1]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal
from tqdm import tqdm, trange


# [2]
data = pd.read_csv("processed_database_2.csv", low_memory=False)
data.reset_index(drop=True, inplace=True)

data.drop([], axis=1)

X = data.iloc[:, 1:].to_numpy()
Y = data.iloc[:, :1].to_numpy()

print(X.shape, Y.shape)


# [3]
# Train/Test Split
# X = X.sample(n=1000).reset_index(drop=True)

idxs = np.random.choice(np.arange(X.shape[0]), int(0.8 * X.shape[0]), replace=False)

X_train = X[idxs]
Y_train = Y[idxs]

X_test = X[~idxs]
Y_test = Y[~idxs]


# [4]
def Xavier_init(n_pre, n_post):
    w0 = np.sqrt(6 / (n_pre + n_post))
    W = np.random.uniform(-w0, w0, (n_post, n_pre))
    b = np.random.uniform(-w0, w0, (n_post, 1))
    return W, b


# [5]
def initialize_weights(n_x: int, n_h: tuple, n_y: int):
    W1, b1 = Xavier_init(n_x, n_h[0])
    W2, b2 = Xavier_init(n_h[0], n_h[1])
    W3, b3 = Xavier_init(n_h[1], n_y)

    We = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2, "b3": b3}
    return We


# [6]
def mse(Y_true: np.ndarray, Y_pred: np.ndarray):
    return ((Y_pred - Y_true) ** 2).mean()


# [7]
def relu(X: np.ndarray):
    A = X * (X > 0)
    dA = 1 * (X > 0)
    return A, dA


# [8]
def linear(X: np.ndarray):
    A = X
    dA = 1
    return A, dA


# [9]
def forward_propagate(We, X):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    Z1 = np.dot(W1, X.T) + b1
    A1, dA1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2, dA2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3, dA3 = linear(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "dA1": dA1,
        "Z2": Z2,
        "A2": A2,
        "dA2": dA2,
        "Z3": Z3,
        "A3": A3,
        "dA3": dA3,
    }
    return cache


# [10]
def calculate_gradients(X: np.ndarray, Y: np.ndarray, We: dict):
    N = X.shape[0]
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    # Forward pass
    cache = forward_propagate(We, X)

    # Calculate loss
    J = mse(Y.T, cache["A3"])

    # Backward pass
    dZ3 = 2 * (cache["A3"] - Y.T) * cache["dA3"]
    dW3 = 1 / N * np.dot(dZ3, cache["A2"].T)
    db3 = 1 / N * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = dA2 * cache["dA2"]  # Derivative of ReLU
    dW2 = 1 / N * np.dot(dZ2, cache["A1"].T)
    db2 = 1 / N * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * cache["dA1"]  # Derivative of ReLU
    dW1 = 1 / N * np.dot(dZ1, X)
    db1 = 1 / N * np.sum(dZ1, axis=1, keepdims=True)

    # Return gradients and loss
    dWe = {"dW1": dW1, "dW2": dW2, "dW3": dW3, "db1": db1, "db2": db2, "db3": db3}
    return J, dWe


# [11]
def update_weights(We, dWe, learning_rate):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    dW1 = dWe["dW1"] * learning_rate
    dW2 = dWe["dW2"] * learning_rate
    dW3 = dWe["dW3"] * learning_rate
    db1 = dWe["db1"] * learning_rate
    db2 = dWe["db2"] * learning_rate
    db3 = dWe["db3"] * learning_rate

    W1 -= dW1
    W2 -= dW2
    W3 -= dW3
    b1 -= db1
    b2 -= db2
    b3 -= db3

    We = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2, "b3": b3}
    return We


# [12]
def update_weights_momentum(We, dWe, mWe, learning_rate, momentum_rate):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    dW1 = dWe["dW1"] * learning_rate + mWe["mW1"] * momentum_rate
    dW2 = dWe["dW2"] * learning_rate + mWe["mW2"] * momentum_rate
    dW3 = dWe["dW3"] * learning_rate + mWe["mW3"] * momentum_rate
    db1 = dWe["db1"] * learning_rate + mWe["mb1"] * momentum_rate
    db2 = dWe["db2"] * learning_rate + mWe["mb2"] * momentum_rate
    db3 = dWe["db3"] * learning_rate + mWe["mb3"] * momentum_rate

    W1 -= dW1
    W2 -= dW2
    W3 -= dW3
    b1 -= db1
    b2 -= db2
    b3 -= db3

    We = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2, "b3": b3}
    mWe = {"mW1": dW1, "mW2": dW2, "mW3": dW3, "mb1": db1, "mb2": db2, "mb3": db3}
    return We, mWe


# [13]
def update_weights_adagrad(We, dWe, mWe, learning_rate, epsilon):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    dW1 = mWe["mW1"] + np.square(dWe["dW1"])
    dW2 = mWe["mW2"] + np.square(dWe["dW2"])
    dW3 = mWe["mW3"] + np.square(dWe["dW3"])
    db1 = mWe["mb1"] + np.square(dWe["db1"])
    db2 = mWe["mb2"] + np.square(dWe["db2"])
    db3 = mWe["mb3"] + np.square(dWe["db3"])

    W1 -= (learning_rate / (np.sqrt(dW1) + epsilon)) * dWe["dW1"]
    W2 -= (learning_rate / (np.sqrt(dW2) + epsilon)) * dWe["dW2"]
    W3 -= (learning_rate / (np.sqrt(dW3) + epsilon)) * dWe["dW3"]
    b1 -= (learning_rate / (np.sqrt(db1) + epsilon)) * dWe["db1"]
    b2 -= (learning_rate / (np.sqrt(db2) + epsilon)) * dWe["db2"]
    b3 -= (learning_rate / (np.sqrt(db3) + epsilon)) * dWe["db3"]

    We = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2, "b3": b3}
    mWe = {"mW1": dW1, "mW2": dW2, "mW3": dW3, "mb1": db1, "mb2": db2, "mb3": db3}
    return We, mWe


# [14]
def update_weights_rmsprop(We, dWe, mWe, learning_rate, epsilon, rho):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    dW1 = (rho * mWe["mW1"]) + ((1 - rho) * np.square(dWe["dW1"]))
    dW2 = (rho * mWe["mW2"]) + ((1 - rho) * np.square(dWe["dW2"]))
    dW3 = (rho * mWe["mW3"]) + ((1 - rho) * np.square(dWe["dW3"]))
    db1 = (rho * mWe["mb1"]) + ((1 - rho) * np.square(dWe["db1"]))
    db2 = (rho * mWe["mb2"]) + ((1 - rho) * np.square(dWe["db2"]))
    db3 = (rho * mWe["mb3"]) + ((1 - rho) * np.square(dWe["db3"]))

    W1 -= (learning_rate / (np.sqrt(dW1) + epsilon)) * dWe["dW1"]
    W2 -= (learning_rate / (np.sqrt(dW2) + epsilon)) * dWe["dW2"]
    W3 -= (learning_rate / (np.sqrt(dW3) + epsilon)) * dWe["dW3"]
    b1 -= (learning_rate / (np.sqrt(db1) + epsilon)) * dWe["db1"]
    b2 -= (learning_rate / (np.sqrt(db2) + epsilon)) * dWe["db2"]
    b3 -= (learning_rate / (np.sqrt(db3) + epsilon)) * dWe["db3"]

    We = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2, "b3": b3}
    mWe = {"mW1": dW1, "mW2": dW2, "mW3": dW3, "mb1": db1, "mb2": db2, "mb3": db3}
    return We, mWe


# [15]
def update_weights_adam(We, dWe, mWe, vWe, learning_rate, epsilon, rho1, rho2):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]

    mW1 = (rho1 * mWe["mW1"]) + ((1 - rho1) * dWe["dW1"])
    mW2 = (rho1 * mWe["mW2"]) + ((1 - rho1) * dWe["dW2"])
    mW3 = (rho1 * mWe["mW3"]) + ((1 - rho1) * dWe["dW3"])
    mb1 = (rho1 * mWe["mb1"]) + ((1 - rho1) * dWe["db1"])
    mb2 = (rho1 * mWe["mb2"]) + ((1 - rho1) * dWe["db2"])
    mb3 = (rho1 * mWe["mb3"]) + ((1 - rho1) * dWe["db3"])

    vW1 = (rho2 * vWe["vW1"]) + ((1 - rho2) * np.square(dWe["dW1"]))
    vW2 = (rho2 * vWe["vW2"]) + ((1 - rho2) * np.square(dWe["dW2"]))
    vW3 = (rho2 * vWe["vW3"]) + ((1 - rho2) * np.square(dWe["dW3"]))
    vb1 = (rho2 * vWe["vb1"]) + ((1 - rho2) * np.square(dWe["db1"]))
    vb2 = (rho2 * vWe["vb2"]) + ((1 - rho2) * np.square(dWe["db2"]))
    vb3 = (rho2 * vWe["vb3"]) + ((1 - rho2) * np.square(dWe["db3"]))

    mW1_hat = mW1 / (1 - rho1)
    mW2_hat = mW2 / (1 - rho1)
    mW3_hat = mW3 / (1 - rho1)
    mb1_hat = mb1 / (1 - rho1)
    mb2_hat = mb2 / (1 - rho1)
    mb3_hat = mb3 / (1 - rho1)

    vW1_hat = vW1 / (1 - rho2)
    vW2_hat = vW2 / (1 - rho2)
    vW3_hat = vW3 / (1 - rho2)
    vb1_hat = vb1 / (1 - rho2)
    vb2_hat = vb2 / (1 - rho2)
    vb3_hat = vb3 / (1 - rho2)

    W1 -= learning_rate * (mW1_hat / (np.sqrt(vW1_hat) + epsilon))
    W2 -= learning_rate * (mW2_hat / (np.sqrt(vW2_hat) + epsilon))
    W3 -= learning_rate * (mW3_hat / (np.sqrt(vW3_hat) + epsilon))
    b1 -= learning_rate * (mb1_hat / (np.sqrt(vb1_hat) + epsilon))
    b2 -= learning_rate * (mb2_hat / (np.sqrt(vb2_hat) + epsilon))
    b3 -= learning_rate * (mb3_hat / (np.sqrt(vb3_hat) + epsilon))

    We = {"W1": W1, "W2": W2, "W3": W3, "b1": b1, "b2": b2, "b3": b3}
    mWe = {"mW1": mW1, "mW2": mW2, "mW3": mW3, "mb1": mb1, "mb2": mb2, "mb3": mb3}
    vWe = {"vW1": vW1, "vW2": vW2, "vW3": vW3, "vb1": vb1, "vb2": vb2, "vb3": vb3}
    return We, mWe, vWe


# [16]
# Define training loop
def train(
    X: np.ndarray,
    Y: np.ndarray,
    We: dict,
    num_epochs: int,
    learning_rate: float,
    batch_size: int = 200,
    validation_split: float = 0.1,
    optimizer: Literal["", "none", "momentum", "adagrad", "rmsprop", "adam"] = "none",
    momentum_rate: float = 0.9,
    epsilon: float = 1e-8,
    rho1: float = 0.9,
    rho2: float = 0.99,
    patience: int = 10,
    min_delta: float = 1e-3,
):
    patience_counter = 0
    best_valid_cost = np.inf
    mWe = {
        "mW1": 0,
        "mW2": 0,
        "mW3": 0,
        "mb1": 0,
        "mb2": 0,
        "mb3": 0,
    }
    vWe = {
        "vW1": 0,
        "vW2": 0,
        "vW3": 0,
        "vb1": 0,
        "vb2": 0,
        "vb3": 0,
    }
    if validation_split is None:
        X_train = X
        Y_train = Y
    else:
        random_idxs = np.random.choice(X.shape[0], (X.shape[0]))
        X = X[random_idxs]
        Y = Y[random_idxs]
        valid_start = int(X.shape[0] * validation_split)
        X_train = X[valid_start:]
        Y_train = Y[valid_start:]
        X_valid = X[:valid_start]
        Y_valid = Y[:valid_start]
    N = X_train.shape[0]
    if batch_size is None:
        batch_size = X_train.shape[0]
        mini_batch_count = 1
    elif isinstance(batch_size, float) and batch_size < 1:
        batch_size = int(N * batch_size)
        mini_batch_count = N // batch_size
    else:
        mini_batch_count = N // batch_size

    cache_train = forward_propagate(We, X_train[:batch_size])
    cost_train = mse(Y_train[:batch_size].T, cache_train["A3"])
    costs_train = [cost_train]

    cache_valid = forward_propagate(We, X_valid[: (batch_size // 10)])
    cost_valid = mse(Y_valid[: (batch_size // 10)].T, cache_valid["A3"])
    costs_valid = [cost_valid]

    for epoch in range(num_epochs):
        cost_train_total = 0
        cost_valid_total = 0

        mini_batch_start = 0
        mini_batch_end = batch_size

        pbar = trange(
            mini_batch_count,
            desc=f"Epoch {epoch+1:4d}/{num_epochs}",
            ncols=130,
            leave=True,
        )
        for _ in pbar:
            mini_batch_x = X_train[mini_batch_start:mini_batch_end]
            mini_batch_y = Y_train[mini_batch_start:mini_batch_end]

            mini_batch_x_valid = X_valid[mini_batch_start // 10 : mini_batch_end // 10]
            mini_batch_y_valid = Y_valid[mini_batch_start // 10 : mini_batch_end // 10]

            cost_train, dWe = calculate_gradients(mini_batch_x, mini_batch_y, We)
            cost_train_total += cost_train
            match optimizer:
                case ["" | "none"]:
                    We = update_weights(We, dWe, learning_rate)
                case "momentum":
                    We, mWe = update_weights_momentum(
                        We, dWe, mWe, learning_rate, momentum_rate
                    )
                case "adagrad":
                    We, mWe = update_weights_adagrad(
                        We, dWe, mWe, learning_rate, epsilon
                    )
                case "rmsprop":
                    We, mWe = update_weights_rmsprop(
                        We, dWe, mWe, learning_rate, epsilon, rho1
                    )
                case "adam":
                    We, mWe, vWe = update_weights_adam(
                        We, dWe, mWe, vWe, learning_rate, epsilon, rho1, rho2
                    )

            cache_valid = forward_propagate(We, mini_batch_x_valid)
            cost_valid = mse(mini_batch_y_valid.T, cache_valid["A3"])
            cost_valid_total += cost_valid

            pbar.set_postfix_str(
                f" Training Error: {cost_train:.5f}, Validation Error: {cost_valid:.5f}",
                refresh=False,
            )

            mini_batch_start = mini_batch_end
            mini_batch_end = min(mini_batch_end + batch_size, N)

        costs_train.append(cost_train_total / mini_batch_count)
        costs_valid.append(cost_valid_total / mini_batch_count)

        if costs_valid[-1] + min_delta < best_valid_cost:
            patience_counter = 0
            best_valid_cost = costs_valid[-1]
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Print loss
        # if (epoch + 1) % 100 == 0:
        # if validation_split is None:
        #    print(f"Epoch: {epoch+1}/{num_epochs} - Training Error: {round(J, 5)}")
        # else:
        #    print(
        #        f"Epoch: {epoch+1}/{num_epochs} - Training Error: {round(J, 5)} - Validation Error: {round(cost_valid, 5)}            ", end="\r"
        #    )

    return We, costs_train, costs_valid


# [17]
# Train the model
We = initialize_weights(15, (200, 200), 1)
We_final, costs_train, costs_valid = train(
    X_train,
    Y_train,
    We,
    num_epochs=1000,
    batch_size=0.01,
    learning_rate=0.001,
    optimizer="adam",
)


# [18]
fig, ax = plt.subplots()
ax.plot(np.arange(len(costs_train)), costs_train, "b", label="Training")
ax.plot(np.arange(len(costs_valid)), costs_valid, "r", label="Validation")
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Iteration vs Cost (Neural Network)")

fig.savefig("neural_cost")


# [19]
cache_train = forward_propagate(We_final, X_train)
Y_pred_train = cache_train["A3"]
residuals_train = Y_train.T - Y_pred_train
train_cost = mse(Y_train.T, Y_pred_train)
print(f"Training cost: {np.squeeze(train_cost).round(2)}")

fig, ax = plt.subplots()
ax.scatter(Y_train, Y_pred_train, (0.01))
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Actual vs Prediction (Neural Network, Train)")

ax.plot(np.arange(100), np.arange(100), "r")
fig.savefig("neural_train_pred")


fig, ax = plt.subplots()
ax.set_autoscale_on(False)
ax.scatter(Y_train, residuals_train, (0.01))
ax.set_xbound(-50, 100)
ax.set_ybound(-100, 100)
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")
ax.set_title("Predicted Value vs Residual (Neural Network, Train)")

ax.plot(np.arange(150) - 50, np.zeros((150,)), "black")

fig.savefig("neural_train_resd")


# [20]
cache_test = forward_propagate(We_final, X_test)
Y_pred_test = cache_test["A3"]
residuals_test = Y_test.T - Y_pred_test
test_cost = mse(Y_test.T, Y_pred_test)
print(f"Training cost: {np.squeeze(test_cost).round(2)}")

fig, ax = plt.subplots()
ax.scatter(Y_test, Y_pred_test, (0.01))
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Actual vs Prediction (Neural Network, Test)")

ax.plot(np.arange(100), np.arange(100), "r")
fig.savefig("neural_test_pred")


fig, ax = plt.subplots()
ax.set_autoscale_on(False)
ax.scatter(Y_pred_test, residuals_test, (0.01))
ax.set_xbound(-50, 100)
ax.set_ybound(-100, 100)
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")
ax.set_title("Predicted Value vs Residual (Neural Network, Test)")

ax.plot(np.arange(150) - 50, np.zeros((150,)), "black")

fig.savefig("neural_test_resd")
