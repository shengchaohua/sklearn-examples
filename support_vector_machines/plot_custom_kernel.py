import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


# Import some data to play with
iris = datasets.load_iris()
X = iris.data[:, :2] # Take the first two features
Y = iris.target


def my_kernel(X, Y):
    """
    Custom kernel
                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    :param X: matrix
    :param Y: matrix
    :return:
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)


h = .02 # step size in the mesh

# Create an instance of SVM and fit out data
clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)

# Plot tge decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]*[y_min, y_max]
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot also the training points
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors='k')
plt.title('3-Class classification using Support Vector Machine with custom'
          ' kernel')
plt.axis('tight')
plt.show()