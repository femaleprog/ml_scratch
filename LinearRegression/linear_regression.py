import numpy as np
class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
     
     #gradient descent   
    def fit(self, X, y):
        # init parameters
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0 
        
        for _ in range(self.n_iters):
            y_predicted = np.dot(X, self.weights) + self.bias
            
            dw = (1/n_samples)*np.dot(X.T, (y_predicted-y)) 
            db = (1/n_samples)*np.sum(y_predicted-y)
            
            # w = w - alpha*dw 
            
            self.weights -= self.lr*dw
            
            # b = b - alpha*db 
            
            self.bias -= self.lr*db
            
    def predict(self, X):
         y_predicted = np.dot(X, self.weights) + self.bias
         return y_predicted
