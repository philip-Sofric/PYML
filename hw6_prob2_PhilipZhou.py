import scipy.io
import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt


# form covariance matrix for input d*n , output is d*d
def covariance_matrix(xarray):
    d_number = len(xarray[:, 0])
    n_number = len(xarray[0, :])
    mu = np.mean(xarray, axis=1)
    S = np.zeros((d_number, d_number))
    for it in range(n_number):
        new_x = xarray[:, it] - mu
        S += np.outer(new_x, new_x)
    return S / n_number


data = scipy.io.loadmat('yalefaces.mat')
data = data['yalefaces']
dimension = 2016
n = 2414
# index = 35
x = data.reshape(dimension, n)
# plt.imshow(data[:, :, index])
s_matrix = covariance_matrix(x)
U, lambdas, UT = np.linalg.svd(s_matrix, full_matrices=True)
# print(lambdas[:10])
sum_lambda = sum(lambdas)
ratio = 0
k = 0
x_scale = np.linspace(0, dimension, dimension)
while ratio < 0.95:
    ratio = sum(lambdas[:k]) / sum_lambda
    k += 1
print(k)
# ratio = np.zeros(dimension)
# for i in range(dimension):
#     ratio[i] = sum(lambdas[:i]) / sum_lambda
# plt.plot(x_scale, lambdas)
# plt.plot(x_scale, ratio)
figure, axis = plt.subplots(4, 5)
x_bar = np.mean(x, axis=1)
x_bar = x_bar.reshape(48, 42)
axis[0, 0].imshow(x_bar)
axis[0, 0].set_title('Eigenface ' + str(0))
for z in range(19):
    image = U[:, z].reshape(48, 42)
    r = (z+1) // 5
    c = (z+1) % 5
    axis[r, c].imshow(image)
    axis[r, c].set_title('Eigenface ' + str(z+1))

plt.show()
