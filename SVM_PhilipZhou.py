import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

# generating a dataset
np.random.seed(2)

x = np.zeros((300, 2))
y = np.zeros(300)
p1 = 0.4
p2 = 0.6
for i in range(300):
    y[i] = np.random.choice([0, 1], 1, p=[p1, p2])
    if y[i] == 1:
        if i % 2 == 0:
            mean1 = [-1, 1]
            cov1 = [[2, 1], [1, 2]]
            x[i, :] = np.random.multivariate_normal(mean1, cov1, 1)
        else:
            mean1 = [3, -6]
            cov1 = [[2, 1], [1, 2]]
            x[i, :] = np.random.multivariate_normal(mean1, cov1, 1)
    else:
        mean2 = [-1, -3]
        cov2 = [[2, 1], [1, 1]]
        x[i, :] = np.random.multivariate_normal(mean2, cov2, 1)

# plt.plot(x[y==0,0], x[y==0,1], 'rx')
# plt.plot(x[y==1,0], x[y==1,1], 'o')
# plt.axis('equal')
# plt.show()

h = 0.02
C = 1.0  # SVM regularization parameter
# svc = svm.SVC(kernel='poly', C=C, degree=3).fit(x, y)
# svc = svm.SVC(kernel='poly', C=C, degree=10).fit(x, y)
# svc = svm.SVC(kernel='rbf', C=C, gamma=0.1).fit(x, y)
svc = svm.SVC(kernel='rbf', C=C, gamma=10).fit(x, y)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()
