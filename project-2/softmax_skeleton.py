from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta):
    dot_products = theta.dot(X.T)
    c = dot_products.max(axis = 0)
    exp = np.exp(dot_products - c)
    return exp / exp.sum(axis = 0) 

def computeCostFunction(X, Y, theta, lambdaFactor):
    n = X.shape[0]
    k = theta.shape[0]
    d = X.shape[1]
    probabilities = computeProbabilities(X, theta)
    sum_part_1 = 0.0
    for i in xrange(n):
        for j in xrange(k):
            if (Y[i] == j):
                sum_part_1 += np.log(probabilities[j][i])
    sum_part_2 = 0.0
    for i in xrange(k):
        for j in xrange(d):
            sum_part_2 += np.square(theta[i][j])
    return ((-1.0 / n) * sum_part_1) + ((lambdaFactor / 2.0) * sum_part_2)

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    n = X.shape[0]
    k = theta.shape[0]
    d = theta.shape[1]
    probabilities = computeProbabilities(X, theta)
    new_theta = np.zeros([k,d])
    for j in xrange(k):
        summation = 0.0
        for i in xrange(n):
            is_label = 1.0 if Y[i] == j else 0.0
            summation += ( X[i] * (is_label - probabilities[j][i]))
        new_theta[j] = theta[j] - (alpha * (((-1.0 / n) * summation) + (lambdaFactor * theta[j])))
    return new_theta 

def softmaxRegression(X, Y, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        print "iteration:", i
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor)
    return theta, costFunctionProgression
    
def getClassification(X, theta):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta)
    return np.argmax(probabilities, axis = 0)

def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()


def computeTestError(X, Y, theta):
    errorCount = 0.
    assignedLabels = getClassification(X, theta)
    return 1 - np.mean(assignedLabels == Y)