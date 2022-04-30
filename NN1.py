from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

X, y = load_digits(n_class=2, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=2)
mlp = MLPClassifier(random_state=2)
mlp.fit(X_train, y_train)
print(mlp.score(X_test, y_test))
y_predict = mlp.predict(X_test)
print(X_test.shape)
x = X_test[1]
# plt.matshow(x.reshape(8, 8), cmap=plt.cm.gray)
plt.imshow(x.reshape(8, 8), cmap='binary')
plt.xticks(())
plt.yticks(())
# plt.axis('off')
plt.show()