# Felix Matathias, em2296
# Data Science Institute
# Columbia University 

import random
import numpy as np

class CrossValidation:
    
    def __init__(self, x, y, k=10, classifier=None, debug=False):
        self.k = k
        self.x = x
        self.y = y
        self.n = x.shape[0]
        self.classifier = classifier
        self.debug = debug
    
    
    def get_cross_validation_error(self):
        
        # Concatenate x and y before shuffling (horizontal stack of x and y)
        # Note: Reshaping y to (n_samples, 1) because the input arrays must
        #       have same number of dimensions for hstack
        x_y = np.hstack((self.x, self.y.reshape((self.n,1)))) 
        np.random.shuffle(x_y) 
    
        # Split concatenated array in k equal parts (row-wise)
        n = self.x.shape[0]
        n_splits = n/self.k
        split_idxs = [ i*n_splits for i in range(1, self.k) ] # Index ranges of splits
        x_y_k = np.split(x_y, split_idxs, axis=0)  # A list of k concatenated (x,y) sub-arrays
        
        error_rates = []
        ndim = self.x.shape[1]
        for k in range(0, self.k):
            
            # Split back to x,y the k-th slice of x_y_k, this will be the validation(test) set
            (x_test_k, y_test_k) = np.hsplit(x_y_k[k], np.array([ndim])) 
            
            # Create the train set for this k-fold
            # First create a list of x_y folds that excludes the current k fold
            train_set = [ x_y_k[i] for i in range(0, self.k) if i != k ]
            # Stack the k-1 sub-arrays
            x_y_train = np.vstack(train_set)
            
            # Split back to x,y
            (x_train_k, y_train_k) = np.hsplit(x_y_train, np.array([ndim])) 
            
            # Train classifier with train data
            self.classifier.fit(x_train_k, y_train_k.ravel()) # Need to ravel again splitted y
            
            # Calculate error
            y_pred_k = self.classifier.predict(x_test_k)
            error_rates.append(self.__calc_error_rate(y_pred_k, y_test_k))
            
        if self.debug : print 'Cross validation error rates: {0}'.format(error_rates)
        return np.average(error_rates)
    
    
    def __calc_error_rate(self, y_pred, y):
        n = len(y_pred)
        n_error = 0
        
        for idx, y_p in enumerate(y_pred): 
            if y_p != y[idx]:
                n_error += 1
    
        return (n_error*1.0)/(n*1.0)
    

