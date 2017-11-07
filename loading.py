import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


def load_dataset():
	X = np.load('numpy/X.npy')
	Y = np.load('numpy/Y.npy')
	num_classes = len(np.unique(Y))
	Y = preprocessing.LabelBinarizer().fit_transform(Y)
	X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.20,random_state=3)
	return X_train, X_test, y_train, y_test,num_classes
