from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class multipleLinearRegression():

  def __init__(self, learning_rate = 0.001, iterations = 50):
    self.lr = learning_rate
    self.iter = iterations
    # self.bias = 0
    self.loss = []

  def MSE(self, X, y, weight):
    """
    Calculate error using square error loss method.
    Param:
    X: Array of independent variable
    y: Array of dependent variable
    weight: Array of Weights 

    Returns:
    loss: Mean Square Error Loss of y and y_pred
    y_pred: array of predicted dependent variable
    """
    y_pred = sum(weight * X)
    loss = ((y_pred-y)**2)/len(y)
    return loss, y_pred

  def updateWeights(self, X, y_pred, y_true, weight, index):
    """
    update weights of whole weights array
    new weight = old weight - lr * partial derivative of multiple linear regression
    Param:
    y_true: array of dependent variable
    weight: array of weights 
    index: index of weight, X and y 

    Returns:
    weight: array of updated weight
    """
    for i in range(X.shape[1]):
      # derivative of loss
      weight[i] -= (self.lr * (y_pred-y_true[index]) * X[index][i])
      # derivative of bias
      # self.bias -= (self.lr * (y_pred-y_true[index]))
    return weight

  def trainModel(self, X, y):
    """
    Param:
    X: Array of independent variable from split_data() in newdf.py
    y: Array of dependent variable from split_data() in newdf.py

    Returns:
    weight: array of trained weights
    """
    num_rows = X.shape[0]                                         # Number of Rows of X array
    num_cols = X.shape[1]                                         # Number of Columns of X array
    weight = np.random.randn(1,num_cols) / np.sqrt(num_rows)      # Xavier weight initialization 

    #Calculating Loss and Updating Weights
    for j in range(1, self.iter+1):                               # loop through iterations stated in self.iter to train model
      totalloss = 0
      for i in [i for i in range(num_rows)]:                      # loop through list of indexes
        loss, y_pred = self.MSE(X[i], y[i], weight[0])
        totalloss += loss
        weight[0] = self.updateWeights(X, y_pred, y, weight[0], i)
      self.loss.append(totalloss)                                 # add total loss of current iter to self.loss. Will use for plotting loss to iter
    return weight[0]

  def testModel(self, X_test, y_test, trained_weights):
    """
    Test trained model using x_test and y_test, returns list of predicted result and err
    Param:
    X_test: Array of independent variable from split_data() in newdf.py
    y_test: Array of dependent variable from split_data() in newdf.py
    trained_weights: Array of trained weights from trainmodel function

    Returns:
    test_pred (list) : Predicted Target Variable
    test_loss (list) : Calculated Sqaured Error Loss for y and y_pred
    """
    test_pred = []
    test_loss = []
    for i in [i for i in range(X_test.shape[0])]:                                   # loop through indexes of X_test array
        loss, y_test_pred = self.MSE(X_test[i], y_test[i], trained_weights)
        test_pred.append(y_test_pred)
        test_loss.append(loss)
    return test_pred, test_loss
    
  def predict(self, trained_weights, X_sample):
    prediction = sum(trained_weights * X_sample)
    return prediction

  def plotLoss(self):
    """
    Plots a graph of Loss to iterations
    Param:
    loss (list) : list of loss per iter
    num_of_iter: list of 1 to len(self.iter)
    """
    num_of_iter = [i for i in range(1,self.iter+1)]
    plt.plot(num_of_iter, self.loss)
    plt.xlabel('Number of Iterations')
    plt.ylabel('Loss')
    plt.title('Convergence of gradient descend')
    plt.show()

  def actual_to_pred_graph(self, Y_test, Y_pred):
    """
    Plots a graph of ytest to ypred
    Param:
    loss (list) : list of loss per iter
    num_of_iter: list of 1 to len(self.iter)
    """
    plt.figure(figsize=(15,10))
    plt.scatter(Y_test, Y_pred)
    plt.xlabel('actual')
    plt.ylabel('predicted')
    plt.title('actual vs predicted')
    plt.show()

  def pred_all(self, trained_weight, X_test, Y_test):
    """
    Predict all data using X and Y_test
    Param:
    trained_weight: array of trained weights
    X_test: array of x variables
    Y_test: array of y variable
    """
    interest = ['storey_range', 'floor_area_sqm', 'flat_type', 'remaining_lease', 'year', 'town']   # columns used for prediction
    allPredicts, allDiff, allAccuracy = [], [], []
    xtestdf = pd.DataFrame(X_test)
    xtestdf.columns = interest
    ytestdf = pd.DataFrame(Y_test)
    ytestdf.columns = ['resale_price']
    for i in range(X_test.shape[0]):                                  # for range in test data, find predicted and difference between 
      X_sample = [xtestdf[key][i] for key in interest]                # pred and actual resaleprice
      resale_price = ytestdf['resale_price'][i]

      pred = self.predict(trained_weight, X_sample)                   # predict value
      allPredicts.append(pred)
      
      diff = abs(resale_price - pred)                                 # difference between actual - pred
      allDiff.append(diff)

      accuracy = abs(100*(resale_price - diff)/resale_price)          # accuracy of prediction
      allAccuracy.append(accuracy)

    xtestdf = xtestdf.join(ytestdf)                                   # add in new columns into df
    xtestdf['predicted'] = allPredicts
    xtestdf['difference'] = allDiff
    xtestdf['accuracy'] = allAccuracy
    xtestdf.to_excel('testpred.xlsx', index = False)