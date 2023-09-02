import numpy as np
from dataloader import cifar10


class KNN(object):
    def __init__(self) -> None:
        pass

    def fit(self,X ,y):
        self.Xtr = X
        self.ytr = y


    def predict(self, X):
        num_test=X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

        return Ypred

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    

    if __name__=='__main__':
        
        x_train, y_train, x_test, y_test=cifar10(100,100)
        
        print("dataset",x_train.shape, y_train.shape)
