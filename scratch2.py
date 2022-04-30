import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

x = np.array([[1,0],[2,-1],[-1,1]])
y = np.array([1,1,-1])
h = 0.02
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(x, y)
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
print(svc.coef_)
Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.coolwarm)
plt.show()