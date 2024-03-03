import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#numpy: a library for scientific computing with Python. It provides a high-performance multidimensional array object, and tools for working with these arrays.
#matplotlib.pyplot: a library for creating static, animated, and interactive visualizations in Python.
#pandas: a library for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series.

from google.colab import drive
drive.mount('/content/drive')

#these lines act as connector to link your code with your google drive



path='/content/drive/MyDrive/Colab Notebooks/Social_Network_Ads.csv'
dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Social_Network_Ads.csv')

X = dataset.iloc[:, [2, 3]].values
#x is independent variable(On what basis we want to classify)
#y is dependent variables(What we want to classify)
y = dataset.iloc[:, 4].values
#2,3,4 are the coloum name which are given as an input

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

from sklearn.preprocessing import StandardScaler
#This line imports the StandardScaler class from the sklearn.preprocessing module. This class is specifically designed to standardize features in a dataset.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
#Fitting: The fit method calculates the mean and standard deviation of each feature in the training set (X_train). It stores these values internally for later use.
#Transforming: The transform method applies the standardization to the training data. It subtracts the mean from each feature and then divides by the standard deviation. This results in the standardized training data being stored in X_train
X_test = sc.transform(X_test)

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
# Instantiate the model to on which algorithm the data should be trained
classifier.fit(X_train, y_train)
# here is the model is being trained by sending the value
#here classifer is an identifer representing the model
#----------------------------------------------------------------
# Predict accuracy of training data
from sklearn.metrics import accuracy_score
y_train_pred = classifier.predict(X_train)
# Calculate accuracy of training data prediction
accuracy_train = accuracy_score(y_train, y_train_pred)

# Print training data accuracy
print("Training data accuracy of the model creating data:", accuracy_train)

#-----------------------------------------------------------------


y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test,y_pred),end='\n')
print("Training Accuracy-----:", accuracy_score)


#A confusion matrix is a table that summarizes the performance of a classification model. It shows the number of correct and incorrect predictions made by the model for each class.

#print(accuracy_score)

from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
