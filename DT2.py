from sklearn.datasets import make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import numpy as np

# X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
X = np.array([[-1.53, -2.86], [-4.42, 0.71], [-1.55, 1.04], [-0.6, -2.01], [-3.43, 1.5],
             [1.45,-1.15], [-1.6, -1.52], [0.79, 0.55], [1.37, -0.23], [1.23, 1.72]])
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
kf = KFold(n_splits=5, shuffle=True, random_state=1)
lr_scores = []
rf_preds = []
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # lr_scores.append(lr.score(X_test, y_test))
    rf = RandomForestClassifier(n_estimators=5, random_state=1)
    rf.fit(X_train, y_train)
    rf_preds.append(rf.predict(X_test))
    # rf_scores.append(rf.score(X_test, y_test))
# print('Linear regression accuracy: ', np.mean(lr_scores))
# print('Random forest accuracy: ', np.mean(rf_scores))
print()
print(rf_preds)