import numpy as np
from dataloader import cifar10


class KNN(object):
    def __init__(self) -> None:
        pass

    def train(self,X ,y):
        self.Xtr = X
        self.ytr = y


    def predict(self, X):
        num_test=X.shape[0]
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            # distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1)
            distances = np.sqrt(np.sum(np.square(self.Xtr - X[i,:]), axis = 1))
            min_index = np.argmin(distances) # get the index with smallest distance
            Ypred[i] = self.ytr[min_index] # predict the label of the nearest example

        return Ypred

    def accuracy(self, y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    

if __name__=='__main__':
    x_train, y_train, x_test, y_test=cifar10(1000,100) # train_dataset = 50000, train_dataset = 10000
    
    print("dataset",x_train.shape, y_train.shape)
    x_train_rows = x_train.reshape(x_train.shape[0], 32 * 32 * 3) # x_train_rows becomes 50000 x 3072
    x_test_rows = x_test.reshape(x_test.shape[0], 32 * 32 * 3) # x_test_rows becomes 10000 x 3072
    print(x_test_rows.shape, x_train_rows.shape)
    nn = KNN() # create a Nearest Neighbor classifier class
    nn.train(x_train_rows, y_train) # train the classifier on the training images and labels
    Yte_predict = nn.predict(x_test_rows) # predict labels on the test images
    # # and now print the classification accuracy, which is the average number
    # # of examples that are correctly predicted (i.e. label matches)
    print(nn.accuracy(y_test, Yte_predict)*100, end="%")