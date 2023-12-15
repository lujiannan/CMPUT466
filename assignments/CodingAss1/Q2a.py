#!/usr/bin/env python3

import pickle as pickle
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None, denormalize=False):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample
    y_hat = None
    loss = None
    risk = None

    y_hat = X.dot(w)
    if (denormalize==True):
        y_hat = y_hat * std_y + mean_y
        y = y * std_y + mean_y

    loss = 1 / (2*len(y)) * (y_hat-y).T.dot(y_hat-y)
    # use np.sum to get a preciser number than sum
    risk = 1 / len(y) * np.sum(np.abs(y_hat-y))

    return y_hat, loss.item(), risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent
            w = w - alpha * 1/len(y_batch) * (X_batch.T).dot(y_hat_batch - y_batch)

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch/batch_size
        losses_train.append(training_loss)
        # 2. Perform validation on the validation set by the risk
        _, _, risk = predict(X_val, w, y_val, denormalize=True)
        risks_val.append(risk)
        # 3. Keep track of the best validation epoch, risk, and the weights
        if risk == np.min(risks_val):
            epoch_best = epoch
            risk_best = risk
            w_best = w

    # Return some variables as needed
    return losses_train, risks_val, epoch_best, risk_best, w_best


############################
# Main code starts here
############################
# Load data
with open("housing.pkl", "rb") as f:
    (X, y) = pickle.load(f)

# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)


# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)

# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay


losses_train, risks_val, epoch_best, risk_best, w_best = train(X_train, y_train, X_val, y_val)
# Perform test by the weights yielding the best validation performance
y_hat, loss, test_risk = predict(X_test, w_best, y_test, denormalize=True)
# Report numbers and draw plots as required.
print(f"Best Epoch: {epoch_best}\nVali Risk: {risk_best}\nTest Risk: {test_risk}")
plt.figure()
plt.plot(range(MaxIter), losses_train)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("q2a_loss.jpg")

plt.figure()
plt.plot(range(MaxIter), risks_val)
plt.xlabel("Epoch")
plt.ylabel("Risk")
plt.savefig("q2a_risk.jpg")
