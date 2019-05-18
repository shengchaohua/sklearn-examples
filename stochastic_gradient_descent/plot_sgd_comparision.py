import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import LogisticRegression


holdout = [0.95, 0.90, 0.75, 0.50, 0.01] # test size
rounds = 20
digits = datasets.load_iris()
X, y = digits.data, digits.target

classifiers = [
    ('SGD', SGDClassifier(max_iter=100, tol=1e-3)),
    ('ASGD', SGDClassifier(average=True, max_iter=1000, tol=1e-3)),
    ('Perceptron', Perceptron(tol=1e-3)),
    ('Passive-Aggressive I', PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0, tol=1e-4)),
    ('Passive-Aggressive II', PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0, tol=1e-4)),
    ('SAG', LogisticRegression(solver='sag', tol=1e-1, C=1.e4/ X.shape[0],
                               multi_class='auto'))
]

xx = 1 - np.array(holdout) # train size

for name, clf in classifiers:
    print('train %s' % name)
    rng = np.random.RandomState(42)
    yy = []
    for hd in holdout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=hd, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

plt.legend(loc="upper right")
plt.xlabel("Proportion train")
plt.ylabel("Test Error Rate")
plt.show()
