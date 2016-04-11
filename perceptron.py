# Felix Matathias, em2296
# Data Science Institute
# Columbia University 

import random
import numpy as np

class AveragedPerceptron:
    
    def __init__(self, epochs=1):
        self.epochs = epochs
    
    
    def fit(self, x, y):
        (self.w, self.b) = self.__fit_averaged_perceptron(x, y, self.epochs)
    
    
    def predict(self, x):
        n = x.shape[0]
        y_pred = []
        prediction = None
        for idx in range(0, n):
            product = np.dot(self.w, x[idx]) + self.b 
            if product > 1e-09:
                prediction = 1
            else:
                prediction = -1
            y_pred.append( prediction )
        return y_pred
    
    
    def __fit_averaged_perceptron(self, x, y, epochs):
        n = x.shape[0]
        ndim = x.shape[1]
        w_avg = np.zeros(ndim, dtype=np.float)
        b_avg = 0.
        # Note: Reshaping y to (n_samples, 1) because the input arrays must 
        #       have same number of dimensions for hstack
        y = y.reshape((n,1))

        x_y = np.hstack((x,y)) # Concatenate x and y before shuffling and split again 
                               # after shuffling
    
        for n_epoch in range(0, epochs):
            np.random.shuffle(x_y)
            (x_s, y_s) = np.hsplit(x_y, np.array([ndim])) # Split back shuffled x, y
            (w,b) = self.__fit_online_perceptron(x_s, y_s)
            w_avg += w
            b_avg += b
    
        return (w_avg*(1./epochs),b_avg*(1./epochs))
    
    
    def __fit_online_perceptron(self, x, y):
        n = x.shape[0]
        ndim = x.shape[1]
        w = np.zeros(ndim, dtype=np.float)
        w_avg = np.zeros(ndim, dtype=np.float)
        b = 0.
        b_avg = 0.
    
        for idx in range(0,n): 
            y_idx = y[idx][0]
            pred = np.dot(w, x[idx]) + b   
            if y_idx * pred <= 1e-09:
                w += y_idx * x[idx]
                b += y_idx
            w_avg += w
            b_avg += b
    
        return(w_avg*(1./n),b_avg*(1./n))

