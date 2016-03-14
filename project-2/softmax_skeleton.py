from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

# rewrite
def computeProbabilities(X, theta):
    n = X.shape[0]
    k = theta.shape[0]
    H = np.zeros([k,n])
    for i in range(n):
        dot_products = np.zeros(k)
        for j in range(k):
            dot_products[j] = np.dot(theta[j], X[i])
        # max_theta_j = np.amax(theta,0)
        c = np.amax(dot_products)
        probabilities = np.zeros(k)
        sum_of_probabilities = 0.0
        for j in range(k):
            probabilities[j] = np.exp(np.dot(theta[j], X[i]) - c)
            sum_of_probabilities += probabilities[j]
        norm_probabilities = (1.0 / sum_of_probabilities) * probabilities
        for j in range(len(norm_probabilities)):
            H[j][i] = norm_probabilities[j]
    return H

def computeCostFunction(X, Y, theta, lambdaFactor):
    n = X.shape[0]
    k = theta.shape[0]
    d = X.shape[1]
    probabilities = computeProbabilities(X, theta)
    sum_part_1 = 0.0
    for i in range(n):
        for j in range(k):
            if (Y[i] == j):
                sum_part_1 += np.log(probabilities[j][i])
    sum_part_2 = 0.0
    for i in range(k):
        for j in range(d):
            sum_part_2 += np.square(theta[i][j])
    return ((-1.0 / n) * sum_part_1) + ((lambdaFactor / 2.0) * sum_part_2)

def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor):
    n = X.shape[0]
    k = theta.shape[0]
    d = theta.shape[1]
    probabilities = computeProbabilities(X, theta)
    new_theta = np.zeros([k,d])
    for j in range(k):
        summation = 0.0
        for i in range(n):
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