import numpy as np

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem X*theta=y using the least squares method
    :param X: numpy input matrix, size [N,m+1] (feature 0 is a column of 1 for bias)
    :param y: numpy input vector, size [N]
    :return theta = (Xt*X)^(-1) * Xt * y: numpy output vector, size [m+1]
    N is the number of samples and m is the number of features=28
  '''
  return np.dot(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: numpy input matrix, size [N,m]
    :param s: numpy input vector of ground truth labels, size [N]
    :return: accuracy of the model = (correct classifications)/(total classifications) type float
    N is the number of samples and m is the number of features=28
  '''
  cl =model.predict(X)
  return (np.count_nonzero((cl-s) == 0)/len(cl) )* 100

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem. length 28
  '''
  return [-0.05333916  ,0.04828803 ,-0.05218676  ,0.00184343 ,-0.02529401 ,-0.01218149
  ,0.09775821 ,-0.00553758 ,-0.01097705 ,-0.02376481  ,0.07476454  ,0.0135223
  ,0.06017403  ,0.14310813  ,0.77540895  ,0.04812822  ,0.04169703 ,-0.02037378
  ,0.03400662 ,-0.00993515  ,0.04642053  ,0.03066783  ,0.00139793 ,-0.04320131
 ,-0.01808249 ,-0.01461981 ,-0.01931981 ,-0.03219853]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value. type float
  '''
  return 1.7853241538938783e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of list of coefficiants for the classification problem.  length 28
  '''
  return [[-3.32430629e-01 ,-3.18665298e-01  ,3.32029683e-01 ,-1.31261574e-01
  ,-2.17517331e-01 ,-3.12756500e-01 ,-1.43163352e-01 ,-2.96724624e-02
  ,-2.06205605e-01  ,4.67787665e-01 ,-7.10442868e-01  ,2.83778925e-03
  ,-5.49904160e-02  ,9.31220366e-01  ,3.29861946e+00 ,-4.75470700e-01
   ,1.67257587e-01  ,5.77209636e-02 ,-1.90965715e-01 ,-1.32619014e-01
   ,1.94951206e-01  ,1.54304738e-01  ,8.87301223e-02 ,-1.47711786e-01
  ,-6.25160411e-01  ,3.09870172e-01 ,-1.07286127e-02  ,5.23347814e-01]]

def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: list with the intercept value. length 1
  '''
  return [0.5111236]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem. length 2.
  '''
  return [0, 1]