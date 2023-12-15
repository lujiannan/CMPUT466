# -*- coding: utf-8 -*-
import numpy as np
import struct
import matplotlib.pyplot as plt
import pandas as pd

# ------------------------
# Report:
# Best Epoch: 40
# Best Accu: 0.9202
# Test Accu: 0.9246
# Train Accu: [0.8902, 0.9006, 0.9061, 0.9092, 0.9104, 0.9117, 0.9122, 0.9125, 0.9134, 0.9143, 0.9147, 0.9152, 0.915, 0.9155, 0.916, 0.9164, 0.9167, 0.9171, 0.9169, 0.9169, 0.9177, 0.9178, 0.9178, 0.9178, 0.9179, 0.9179, 0.918, 0.9183, 0.9183, 0.9182, 0.9184, 0.9185, 0.9183, 0.9186, 0.919, 0.9192, 0.9192, 0.9197, 0.9199, 0.9199, 0.9202, 0.9201, 0.9201, 0.9199, 0.92, 0.9197, 0.9199, 0.92, 0.9201, 0.92]
# ------------------------

def readMNISTdata():
    with open('t10k-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        test_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_data = test_data.reshape((size, nrows*ncols))

    with open('t10k-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        test_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        test_labels = test_labels.reshape((size, 1))

    with open('train-images-idx3-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        train_data = np.fromfile(f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_data = train_data.reshape((size, nrows*ncols))

    with open('train-labels-idx1-ubyte', 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        train_labels = np.fromfile(
            f, dtype=np.dtype(np.uint8).newbyteorder('>'))
        train_labels = train_labels.reshape((size, 1))

    # augmenting a constant feature of 1 (absorbing the bias term)
    train_data = np.concatenate(
        (np.ones([train_data.shape[0], 1]), train_data), axis=1)
    test_data = np.concatenate(
        (np.ones([test_data.shape[0], 1]),  test_data), axis=1)
    _random_indices = np.arange(len(train_data))
    np.random.shuffle(_random_indices)
    train_labels = train_labels[_random_indices]
    train_data = train_data[_random_indices]

    X_train = train_data[:50000] / 256
    t_train = train_labels[:50000]

    X_val = train_data[50000:] / 256
    t_val = train_labels[50000:]

    return X_train, t_train, X_val, t_val, test_data / 256, test_labels

# refer to https://stackoverflow.com/a/39558290
def softmax(y):
    # y = y - np.max(y)
    # return np.exp(y) / np.sum(np.exp(y), axis=0)

    s = np.max(y, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_y = np.exp(y - s)
    div = np.sum(e_y, axis=1)
    div = div[:, np.newaxis] # dito
    return e_y / div

# refer to sklearn.preprocessing OneHotEncoder & https://blog.csdn.net/xiaoyw71/article/details/121981847
def oneHotEncoder(t):
    nb_classes = 10
    targets = t.reshape(-1)
    return np.eye(nb_classes)[targets]


def predict(X, W, t=None):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    # TODO Your code here
    y = X.dot(W)
    y_hat = softmax(y)
    t_hat = np.argmax(y_hat, axis=1)

    m = X.shape[0]
    loss = -1 / m * np.sum(oneHotEncoder(t) * np.log(y_hat))
    acc = np.sum(t_hat == t.flatten()) / len(t_hat)

    gradient = -1 / m * X.T.dot(oneHotEncoder(t) - y_hat)

    return y, t_hat, loss, acc, gradient


def train(X_train, y_train, X_val, t_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # TODO Your code here
    w = np.zeros([X_train.shape[1], N_class])

    losses_train = []
    accs_val = []

    w_best = None
    epoch_best = 0
    acc_best = 0

    for epoch in range(MaxEpoch):

        loss_this_epoch = 0
        for b in range(int(np.ceil(N_train/batch_size))):

            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            _, _, loss_batch, _, gradient = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch

            # Mini-batch gradient descent
            w = w - alpha * gradient - alpha * decay * w

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        training_loss = loss_this_epoch/int(np.ceil(N_train/batch_size))
        losses_train.append(training_loss)
        # 2. Perform validation on the validation set by the acc
        _, _, _, acc, _ = predict(X_val, w, t_val)
        accs_val.append(acc)
        # 3. Keep track of the best validation epoch, acc, and the weights
        if acc == np.max(accs_val):
            epoch_best = epoch
            acc_best = acc
            W_best = w

    # Return some variables as needed
    return epoch_best, acc_best,  W_best, losses_train, accs_val

##############################
# Main code starts here
X_train, t_train, X_val, t_val, X_test, t_test = readMNISTdata()


print(X_train.shape, t_train.shape, X_val.shape,
      t_val.shape, X_test.shape, t_test.shape)


N_class = 10

alpha = 0.1      # learning rate
batch_size = 100    # batch size
MaxEpoch = 50        # Maximum epoch
decay = 0.          # weight decay


# TODO: report 3 number, plot 2 curves
epoch_best, acc_best,  W_best, train_losses, valid_accs = train(X_train, t_train, X_val, t_val)

print("Predicting...")
_, _, _, acc_test, _ = predict(X_test, W_best, t_test)

print("Report:\n{}".format(f"Best Epoch: {epoch_best} \nBest Accu: {acc_best} \nTest Accu: {acc_test} \nTrain Accu: {valid_accs}"))

file_name="accuracy.png"
fig = plt.figure()
plt.plot(valid_accs, label="Validation Accuracy")
plt.title("Validation Accuracy vs. Epoch")
plt.legend()
plt.xlabel('Epoch #')
plt.ylabel('Vali. Accu.')
plt.savefig(file_name)

file_name="loss.png"
fig = plt.figure()
plt.plot(train_losses, label="Training Losses")
plt.title("Training Losses vs. Epoch")
plt.legend()
plt.xlabel('Epoch #')
plt.ylabel('Train Loss')
plt.savefig(file_name)