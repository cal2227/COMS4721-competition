#!/usr/bin/env python

# Felix Matathias, em2296
# Data Science Institute
# Columbia University 

import operator
import numpy as np
from scipy.io import loadmat

# Classifiers implemented in sklearn
# Sources:
# LDA, QDA: http://scikit-learn.org/stable/modules/lda_qda.html
# Logistic: http://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA

# Classifiers and methods implemented by Felix Matathias
from perceptron import AveragedPerceptron
from cross_validation import CrossValidation


def calc_error_rate(y_pred, y):
    n = len(y_pred)
    n_error = 0
    
    for idx, y_p in enumerate(y_pred): 
        if y_p != y[idx]:
            n_error += 1
    
    return (n_error*1.0)/(n*1.0)

# Transforms input feature vectors to 1710-dim space
def transformation_perm_1710(x):
    n = len(x)
    perms = []
    for i in range(0,n):
        for j in range(i+1, n):
            perms.append(x[i]*x[j])
    x_perms = np.array(perms)
    return np.hstack((x, x*x, x_perms))


def get_transformed_data(x, transformation):
    x_trans = np.apply_along_axis(transformation, 1, x)
    return x_trans


def main():

    spam_data = loadmat('spam_fixed.mat')

    # Data preparation
    # Note: y labels must be of shape (n_samples, ) for sklear classifiers
    #       hence we use ravel(). 
    x_train = spam_data['data']
    y_train = spam_data['labels'].ravel()
    x_test  = spam_data['testdata']
    y_test  = spam_data['testlabels'].ravel()
    x_train_trans = get_transformed_data(x_train, transformation_perm_1710)
    x_test_trans  = get_transformed_data(x_test, transformation_perm_1710)

    # Print data logistics
    n_x_train = x_train.shape[0]
    n_x_test  = x_test.shape[0]
    dim = x_train.shape[1]
    print 'Number of training messages: ', n_x_train
    print 'Number of dimensions: ', dim 
    print 'Number of test messages:', n_x_test

    # Define classifiers
    perceptron_classifier = AveragedPerceptron(epochs=64)
    logistic_classifier =  LogisticRegression(penalty='l2', C=10000000.0, fit_intercept=True, intercept_scaling=1)
    lda_classifier = LDA()
    qda_classifier = QDA()
    
    # Map classifiers
    CLASS_PERCEPTRON       = 'perceptron'
    CLASS_LOGISTIC         = 'logistic'
    CLASS_LDA              = 'lda'
    CLASS_QDA              = 'qda'
    CLASS_PERCEPTRON_TRANS = 'perceptron_transformed'
    CLASS_LOGISTIC_TRANS   = 'logistic_transformed'
    
    classifier_list = [CLASS_PERCEPTRON,
                       CLASS_LOGISTIC, 
                       CLASS_LDA,
                       CLASS_QDA,
                       CLASS_PERCEPTRON_TRANS,
                       CLASS_LOGISTIC_TRANS]
    
    classifier = {}
    classifier[CLASS_PERCEPTRON]       = perceptron_classifier 
    classifier[CLASS_LOGISTIC]         = logistic_classifier
    classifier[CLASS_LDA]              = lda_classifier
    classifier[CLASS_QDA]              = qda_classifier
    classifier[CLASS_PERCEPTRON_TRANS] = perceptron_classifier
    classifier[CLASS_LOGISTIC_TRANS]   = logistic_classifier
    
    data = {}
    data[CLASS_PERCEPTRON]       = (x_train,       y_train, x_test,       y_test)
    data[CLASS_LOGISTIC]         = (x_train,       y_train, x_test,       y_test)
    data[CLASS_LDA]              = (x_train,       y_train, x_test,       y_test)
    data[CLASS_QDA]              = (x_train,       y_train, x_test,       y_test)
    data[CLASS_PERCEPTRON_TRANS] = (x_train_trans, y_train, x_test_trans, y_test)
    data[CLASS_LOGISTIC_TRANS]   = (x_train_trans, y_train, x_test_trans, y_test)
    
    # Run 10-fold cross validation for all classifiers
    error_rates = {}
    k = 10
    for cl in classifier_list:
        print "Cross validating classifier: {}".format(cl)
        cross_val = CrossValidation(data[cl][0],
                                    data[cl][1],
                                    k=k,
                                    classifier=classifier[cl],
                                    debug=True)
        error_rates[cl] = cross_val.get_cross_validation_error()

    # Print cross validation error rates 
    for cl, err in error_rates.iteritems():
        print "Classifier {0} has cross validation error: {1}".format(cl, err)

    # Select classifier with smaller error rate
    sorted_error_rates = sorted(error_rates.items(), key=operator.itemgetter(1))
    best_cl_name  = sorted_error_rates[0][0]
    best_cl_error = sorted_error_rates[0][1]
    print "Classifier with smallest error is: {0}  with error rate: {1}".format(best_cl_name,
                                                                                best_cl_error)
    
    # Train the winner with the entire train data set 
    best_classifier = classifier[best_cl_name]
    best_classifier.fit(data[best_cl_name][0], data[best_cl_name][1])

    # Train error rate
    y_pred_train = best_classifier.predict(data[best_cl_name][0])
    best_err_rate_train = calc_error_rate(y_pred_train, data[best_cl_name][1]) 
    print "Best train error rate achieved: {0}".format(best_err_rate_train)
  
    # Test error rate
    y_pred_test = best_classifier.predict(data[best_cl_name][2])
    best_err_rate_test = calc_error_rate(y_pred_test, data[best_cl_name][3])
    print "Best test error rate achieved: {0}".format(best_err_rate_test)

main()


'''
Takes about 15 minutes to run on a MacBook pro Core i7 with 16Gbytes of RAM
Logistic regression in expanded feature space takes the overwhelming majority of time, while
all other classifiers finish within a few seconds.

Results:

Classifier lda has cross validation error: 0.110900952021
Classifier logistic has cross validation error: 0.0779679717546
Classifier qda has cross validation error: 0.159593762478
Classifier perceptron has cross validation error: 0.0835025114011
Classifier perceptron_transformed has cross validation error: 0.130145219931
Classifier logistic_transformed has cross validation error: 0.0870605048021
Classifier with smallest error is: logistic  with error rate: 0.0779679717546
Best train error rate achieved: 0.0707993474715
Best test error rate achieved: 0.076171875
'''
