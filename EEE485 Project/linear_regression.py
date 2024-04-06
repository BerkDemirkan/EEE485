# [1]
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# [2]
np.set_printoptions(precision=3, suppress=True)


# [3]
# Read Data
data = pd.read_csv("processed_database_2.csv", low_memory=False)

# Features List
features = data.iloc[:, 1:].columns.tolist()

data_train = data.sample(frac=0.8)
data_test = data.drop(index=data_train.index)

X_train = data_train.iloc[:, 1:].reset_index(drop=True)
X_test = data_test.iloc[:, 1:].reset_index(drop=True)

X_train.insert(0, "ones", np.ones((X_train.shape[0], 1)))
X_test.insert(0, "ones", np.ones((X_test.shape[0], 1)))

Y_train = data_train.iloc[:, :1].values
Y_test = data_test.iloc[:, :1].values

beta = np.zeros([1, len(features) + 1])

print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape, beta.shape)


# [4]
def mse(Y_true, Y_pred):
    mse_ = np.mean((Y_true - Y_pred) ** 2)
    return mse_


# [5]
def gradient_descent(
    X_train, Y_train, beta, iterations, learning_rate, patience=10, min_delta=1e-3
):
    n = len(X_train)
    best_cost = np.inf
    patience_counter = 0

    costs = []
    for i in range(iterations):
        Y_pred = np.dot(X_train, beta.T)

        cost = mse(Y_train, Y_pred)
        costs.append(cost)

        gradient = (1 / n) * np.dot(X_train.T, (Y_pred - Y_train))

        beta -= learning_rate * gradient.T

        if (i + 1) % 100 == 0:
            print(f"Iteration: {i+1}, Training Cost: {round(cost, 5)}")

        if cost + min_delta < best_cost:
            patience_counter = 0
            best_cost = cost
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stop at iteration {i}. Training Cost: {round(cost, 5)}")
                break

    return beta, costs


# [6]
final_beta, costs = gradient_descent(
    X_train, Y_train, beta, iterations=10000, learning_rate=0.001
)
print(f"Final beta: \n{final_beta}")


# [7]
fig, ax = plt.subplots()
ax.plot(np.arange(len(costs)), costs, "r")
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Iteration vs Cost (Linear Regression)")

fig.savefig("linear_cost")


# [8]
Y_pred_train = np.dot(X_train, final_beta.T)
residuals_train = Y_train - Y_pred_train
train_cost = mse(Y_train, Y_pred_train)
print(f"Training cost: {np.squeeze(train_cost).round(2)}")

fig, ax = plt.subplots()
ax.scatter(Y_train, Y_pred_train, (0.01))
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Actual vs Prediction (Linear Regression, Train)")

ax.plot(np.arange(100), np.arange(100), "r")
fig.savefig("linear_train_pred")


fig, ax = plt.subplots()
ax.set_autoscale_on(False)
ax.scatter(Y_train, residuals_train, (0.01))
ax.set_xbound(-50, 100)
ax.set_ybound(-100, 100)
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")
ax.set_title("Predicted Value vs Residual (Linear Regression, Train)")

ax.plot(np.arange(150) - 50, np.zeros((150,)), "black")

fig.savefig("linear_train_resd")


# [9]
Y_pred_test = np.dot(X_test, final_beta.T)
residuals_test = Y_test - Y_pred_test
test_cost = mse(Y_test, Y_pred_test)
print(f"Training cost: {np.squeeze(test_cost).round(2)}")

fig, ax = plt.subplots()
ax.scatter(Y_test, Y_pred_test, (0.01))
ax.set_xlabel("Actual Value")
ax.set_ylabel("Predicted Value")
ax.set_title("Actual vs Prediction (Linear Regression, Test)")

ax.plot(np.arange(100), np.arange(100), "r")
fig.savefig("linear_test_pred")


fig, ax = plt.subplots()
ax.set_autoscale_on(False)
ax.scatter(Y_pred_test, residuals_test, (0.01))
ax.set_xbound(-50, 100)
ax.set_ybound(-100, 100)
ax.set_xlabel("Predicted Value")
ax.set_ylabel("Residual")
ax.set_title("Predicted Value vs Residual (Linear Regression, Test)")

ax.plot(np.arange(150) - 50, np.zeros((150,)), "black")

fig.savefig("linear_test_resd")
