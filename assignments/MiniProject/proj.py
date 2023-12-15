import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import hamming_loss
from sklearn.neighbors import KNeighborsClassifier

'''
Logistic Regression Report:
Best Penalty: l1 Best Alpha: 0.00016 Best Accu: 0.9184680298604351 Test Accu: 0.8882978723404256

CVM Report:
Best Hyperparam: 50 Best Accu: 0.9209183673469389 Test Accu: 0.9308510638297872

KNN Report:
Best Neighbor: 21 Best Accu: 0.8928571428571429 Test Accu: 0.8882978723404256
'''

def readData():
    df = pd.read_csv('balance-scale.data')
    columnNames = ['className','leftWeight','leftDistance','rightWeight','rightDistance']
    df.columns = columnNames
    x = df.drop(columns='className')
    y = df['className']
    # mapping labels from LRB to 012
    labelMapping = {'L': 0, 'R': 1, 'B': 2}
    y = y.map(labelMapping)

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3, random_state=33)
    xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, test_size=0.1, random_state=33)
    
    return xTrain, xTest, xVal, yTrain, yTest, yVal

def trainLogisticRegression(xTrain, xTest, xVal, yTrain, yTest, yVal):
    # train
    iterNum = 10
    alpha = 0.1
    penalties = ['l1', 'l2', 'elasticnet', None]

    accsVals = []
    accsL1 = []
    accsL2 = []
    accsElNet = []
    accsNone = []
    bestEpoch = 0
    bestPenalty = ''
    bestAlpha = 0.1
    bestAccuracy = 1

    for epoch in range(iterNum):
        for penalty in penalties:
            while alpha >= 0.0001:
                # different loss equations are tested already, best (stabler) one is log_loss, so use log_loss here to reduce bias
                model = SGDClassifier(loss='log_loss', penalty=penalty, alpha=alpha, max_iter=1000)
                model.fit(xTrain, yTrain)
                # cross-validation
                accuracy = np.mean(cross_val_score(model, xTrain, yTrain, scoring='accuracy'))
                accsVals.append(accuracy)

                if (penalty == 'l1'):
                    accsL1.append(accuracy)
                elif (penalty == 'l2'):
                    accsL2.append(accuracy)
                elif (penalty == 'elasticnet'):
                    accsElNet.append(accuracy)
                elif (penalty == None):
                    accsNone.append(accuracy)

                # track the great accuracy
                if accuracy == np.max(accsVals):
                    bestEpoch = epoch
                    bestPenalty = penalty
                    bestAlpha = alpha
                    bestAccuracy = accuracy
                # alpha value decreases by 1/2 each loop til 0.0001
                alpha = alpha / 5
            # reset alpha value
            alpha = 0.1

    model = SGDClassifier(loss='log_loss', penalty=bestPenalty, alpha=bestAlpha, max_iter=1000)
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    accTest = accuracy_score(yTest, yHat)
    print("Logistic Regression Report:\n{}".format(f"Best Penalty: {bestPenalty} Best Alpha: {bestAlpha} Best Accu: {bestAccuracy} Test Accu: {accTest}"))

    file_name="LogisticRregressionAccuracy.png"
    fig = plt.figure()
    plt.plot(accsL1, label="Accuracy with l1 penalty")
    plt.plot(accsL2, label="Accuracy with l2 penalty")
    plt.plot(accsElNet, label="Accuracy with elasticnet penalty")
    plt.plot(accsNone, label="Accuracy with no penalty")
    plt.title("Logistic Regression Validation Accuracy")
    plt.legend()
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.savefig(file_name)

def trainSVM(xTrain, xTest, xVal, yTrain, yTest, yVal):
    yTrain = yTrain.values.reshape((len(yTrain),))
    hyperparams = [0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 500]
    bestHP = 0
    accsVals = []
    lossVals = []
    bestAccuracy = 0
    MSErrs = []

    for hp in hyperparams:
        svm = SVC(C=hp)
        # cross validation accuracy
        accuracy = np.mean(cross_val_score(svm, xTrain, yTrain, scoring='accuracy', cv=7))
        accsVals.append(accuracy)
        # negative MSE
        MSErr = np.mean(cross_val_score(svm, xTrain, yTrain, scoring='neg_mean_squared_error', cv=7))
        MSErrs.append(MSErr)
        # loss
        svm.fit(xTrain, yTrain)
        yHat = svm.predict(xTrain)
        lossVal = hamming_loss(yTrain, yHat)
        lossVals.append(lossVal)

        # track the great accuracy
        if accuracy == np.max(accsVals):
            bestHP = hp
            bestAccuracy = accuracy

    model = SVC(C=bestHP)
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    accTest = accuracy_score(yTest, yHat)
    print("CVM Report:\n{}".format(f"Best Hyperparam: {bestHP} Best Accu: {bestAccuracy} Test Accu: {accTest}"))

    file_name="SVMAccuracy.png"
    fig = plt.figure()
    plt.plot(np.log(hyperparams), accsVals, label="Accuracy with hyperparams (ln)")
    plt.title("SVM Validation Accuracy")
    plt.legend()
    plt.xlabel('Hyperparam')
    plt.ylabel('Accuracy')
    plt.savefig(file_name)

    file_name="SVMMSE.png"
    fig = plt.figure()
    plt.plot(np.log(hyperparams), MSErrs, label="MSE with hyperparams (ln)")
    plt.title("SVM Mean Squared Error")
    plt.legend()
    plt.xlabel('Hyperparam')
    plt.ylabel('MSE')
    plt.savefig(file_name)

    file_name="SVMLoss.png"
    fig = plt.figure()
    plt.plot(np.log(hyperparams), lossVals, label="Loss with hyperparams (ln)")
    plt.title("SVM Loss")
    plt.legend()
    plt.xlabel('Hyperparam')
    plt.ylabel('Loss')
    plt.savefig(file_name)

def KNNTraining(xTrain, xTest, xVal, yTrain, yTest, yVal):
    yTrain = yTrain.values.reshape((len(yTrain),))
    neighbors = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25]
    bestN = 0
    accsVals = []
    lossVals = []
    bestAccuracy = 0
    MSErrs = []

    for hp in neighbors:
        knn = KNeighborsClassifier(n_neighbors=hp)
        # cross validation accuracy
        accuracy = np.mean(cross_val_score(knn, xTrain, yTrain, scoring='accuracy', cv=7))
        accsVals.append(accuracy)
        # negative MSE
        MSErr = np.mean(cross_val_score(knn, xTrain, yTrain, scoring='neg_mean_squared_error', cv=7))
        MSErrs.append(MSErr)
        # loss
        knn.fit(xTrain, yTrain)
        yHat = knn.predict(xTrain)
        lossVal = hamming_loss(yTrain, yHat)
        lossVals.append(lossVal)

        # track the great accuracy
        if accuracy == np.max(accsVals):
            bestN = hp
            bestAccuracy = accuracy

    model = KNeighborsClassifier(n_neighbors=hp)
    model.fit(xTrain, yTrain)
    yHat = model.predict(xTest)
    accTest = accuracy_score(yTest, yHat)
    print("KNN Report:\n{}".format(f"Best Neighbor: {bestN} Best Accu: {bestAccuracy} Test Accu: {accTest}"))

    file_name="KNNAccuracy.png"
    fig = plt.figure()
    plt.plot(np.log(neighbors), accsVals, label="Accuracy with neighbors (ln)")
    plt.title("KNN Validation Accuracy")
    plt.legend()
    plt.xlabel('Neighbor')
    plt.ylabel('Accuracy')
    plt.savefig(file_name)

    file_name="KNNMSE.png"
    fig = plt.figure()
    plt.plot(np.log(neighbors), MSErrs, label="MSE with neighbors (ln)")
    plt.title("KNN Mean Squared Error")
    plt.legend()
    plt.xlabel('Neighbor')
    plt.ylabel('MSE')
    plt.savefig(file_name)

    file_name="KNNLoss.png"
    fig = plt.figure()
    plt.plot(np.log(neighbors), lossVals, label="Loss with neighbors (ln)")
    plt.title("KNN Loss")
    plt.legend()
    plt.xlabel('Neighbor')
    plt.ylabel('Loss')
    plt.savefig(file_name)

def main():
    xTrain, xTest, xVal, yTrain, yTest, yVal = readData()
    trainLogisticRegression(xTrain, xTest, xVal, yTrain, yTest, yVal)
    trainSVM(xTrain, xTest, xVal, yTrain, yTest, yVal)
    KNNTraining(xTrain, xTest, xVal, yTrain, yTest, yVal)

if __name__ == '__main__':
    main()