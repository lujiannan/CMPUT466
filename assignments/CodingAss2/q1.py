from utils import plot_data, generate_data
import numpy as np


"""
Documentation:

Function generate() takes as input "A" or "B", it returns X, t.
X is two dimensional vectors, t is the list of labels (0 or 1).    

Function plot_data(X, t, w=None, bias=None, is_logistic=False, figure_name=None)
takes as input paris of (X, t) , parameter w, and bias. 
If you are plotting the decision boundary for a logistic classifier, set "is_logistic" as True
"figure_name" specifies the name of the saved diagram.
"""

# -----------------------------------------------------
# Accuracy of linear regression on dataset A: 0.61
# Accuracy of logistic regression on dataset A: 0.9325
# Accuracy of linear regression on dataset B: 0.5
# Accuracy of logistic regression on dataset B: 0.9425
# -----------------------------------------------------

def train_logistic_regression(X, t):
    """
    Given data, train your logistic classifier.
    Return weight and bias
    """
    m = X.shape[1]
    n = X.shape[0]
    w = np.zeros(m)
    b = 0
    for i in range(iterNum):
        y_hat = 1 / (1 + np.exp(-(X.dot(w) + b)))
        w = w - 1 / m * alpha * X.T.dot(y_hat - t)
        b = b - 1 / m * alpha * np.sum(y_hat - t)

    return w, b


def predict_logistic_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    # if-else is not working here (don't know why), reference: stackflow
    t = (1 / (1 + np.exp(-(X @ w + b)))) >= 0.5

    return t


def train_linear_regression(X, t):
    """
    Given data, train your linear regression classifier.
    Return weight and bias
    """
    m = X.shape[1]
    n = X.shape[0]
    w = np.zeros(m)

    ex = np.ones((n,1))
    X = np.append(X, ex, axis=1)
    w = np.linalg.inv((X.T).dot(X)).dot(X.T).dot(t)

    b = w[m]
    w = w[0:m]

    return w, b


def predict_linear_regression(X, w, b):
    """
    Generate predictions by your logistic classifier.
    """
    t = (X.dot(w) + b) >= 0

    return t


def get_accuracy(t, t_hat):
    """
    Calculate accuracy,
    """
    # correct / total
    acc = np.sum(t == t_hat) / len(t)

    return acc


def main():
    global iterNum, alpha
    iterNum = 10000
    alpha = 0.1

    # Dataset A
    # Linear regression classifier
    X, t = generate_data("A")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_A_linear.png')

    # logistic regression classifier
    X, t = generate_data("A")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset A:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_A_logistic.png')

    # Dataset B
    # Linear regression classifier
    X, t = generate_data("B")
    w, b = train_linear_regression(X, t)
    t_hat = predict_linear_regression(X, w, b)
    print("Accuracy of linear regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=False,
              figure_name='dataset_B_linear.png')

    # logistic regression classifier
    X, t = generate_data("B")
    w, b = train_logistic_regression(X, t)
    t_hat = predict_logistic_regression(X, w, b)
    print("Accuracy of logistic regression on dataset B:", get_accuracy(t_hat, t))
    plot_data(X, t, w, b, is_logistic=True,
              figure_name='dataset_B_logistic.png')


main()
