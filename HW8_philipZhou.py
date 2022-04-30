import csv
import numpy as np
from numpy import linalg
from numpy import random
import matplotlib.pyplot as plt

with open('EM.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    data1 = []
    for row in csv_reader:
        data1.append(row)
data1 = np.array(data1)


# print(data1.shape)


# probability density function
# x_data is 1*d, mu_data is 1*d, sigma_array is d*d, returned probability is a real value
def pdf(x_data, mu_data, sigma_array):
    dim = len(x_data)
    p1 = np.power(2 * np.pi, -dim / 2)
    det = linalg.det(sigma_array)
    p2 = np.power(det, -1 / 2)
    inv = linalg.inv(sigma_array)
    p3 = x_data - mu_data
    p4 = np.dot(p3, inv)
    p5 = np.dot(p4, p3)
    p6 = np.exp(-1 / 2 * p5)
    return p1 * p2 * p6


# Calculate objective function
# x_array is n*d, w_matrix is n*k, phi_array is K*1, mu_array is K*1, sigma_matrix is k*d*d
def LB(x_array, w_matrix, phi_array, mu_array, sigma_matrix):
    s_sum = 0
    num = len(x_array[:, 0])
    K_value = len(w_matrix[0])
    for ii in range(num):
        for kk in range(K_value):
            prob_0 = pdf(x_array[ii], mu_array[kk], sigma_matrix[kk]) * phi_array[kk]
            t_0 = np.log(prob_0 / w_matrix[ii, kk])
            s_sum += w_matrix[ii, kk] * t_0
    return s_sum


dimension = len(data1[0])
n = len(data1[:, 0])
K = 3
itri = 0
phi_itri = [1 / K, 1 / K, 1 / K]
random_initial = random.randint(0, n, size=K)
mu_itri = [data1[random_initial[0]], data1[random_initial[1]], data1[random_initial[2]]]
data2 = np.transpose(data1)
sigma_itri = [np.matmul(data2, data1), np.matmul(data2, data1), np.matmul(data2, data1)]
w = np.zeros((n, K))
while itri <= 50:
    for i in range(n):
        prob = np.zeros(K)
        for k in range(K):
            prob[k] = pdf(data1[i], mu_itri[k], sigma_itri[k]) * phi_itri[k]
        for k in range(K):
            w[i, k] = prob[k] / sum(prob)

    lb = LB(data1, w, phi_itri, mu_itri, sigma_itri)
    # print('Iteration # is ', itri)
    # print(lb)
    for k in range(K):
        phi_itri[k] = np.mean((w[:, k]))
        s0 = 0
        s1 = 0
        for i in range(n):
            cen_data = data1[i] - mu_itri[k]
            s0 += w[i, k] * np.outer(cen_data, cen_data)
            s1 += w[i, k] * data1[i]
        mu_itri[k] = s1 / sum(w[:, k])
        sigma_itri[k] = s0 / sum(w[:, k])
    itri += 1
print('phi vector is :', phi_itri)
print('mu vector is ', mu_itri)
print('Sigma matrices are', sigma_itri)
ax = plt.axes(projection='3d')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('w')
# ax.scatter(data1[:, 0], data1[:, 1], w[:, 0], marker='x', label=' 1 distribution')
# ax.scatter(data1[:, 0], data1[:, 1], w[:, 1], marker='o', label=' 2 distribution')
# ax.scatter(data1[:, 0], data1[:, 1], w[:, 2], marker='s', label=' 3 distribution')
X1 = np.linspace(2, 12, 100)
X2 = np.linspace(2, 12, 100)
data = np.zeros(2)

for k in range(3):
    X3 = []
    for count in range(100):
        data[0] = X1[count]
        data[1] = X2[count]
        data = np.transpose(data)
        X3.append(pdf(data, mu_itri[k], sigma_itri[k]))
    ax.scatter(X1, X2, X3, marker='s', label=' distribution')
plt.legend()
plt.show()
