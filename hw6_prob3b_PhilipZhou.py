import csv
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import math


# calculate distance between two points; input points are 1-D arrays with same dimension
def distance(point1, point2):
    dim = len(point1)
    sum_dis = 0
    for index in range(dim):
        sum_dis += (point1[index] - point2[index]) ** 2
    return np.sqrt(sum_dis)


data1 = []
with open('data2.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
    for row in csv_reader:
        data1.append(row)
data1 = np.array(data1)
# plt.plot(data1[:, 0], data1[:, 1], 'o')
n = len(data1[:, 0])
# modification on raw data
new_data = np.zeros((n, 2))
for i in range(n):
    new_data[i, 0] = np.sqrt(data1[i, 0] ** 2 + data1[i, 1] ** 2)
    if data1[i, 0] >= 0 and new_data[i, 0] != 0:
        new_data[i, 1] = math.asin(data1[i, 1] / new_data[i, 0])
    elif data1[i, 0] < 0 and new_data[i, 0] != 0:
        new_data[i, 1] = math.pi - math.asin(data1[i, 1] / new_data[i, 0])
    else:
        new_data[i, 1] = 0
# plt.plot(data1[:, 0], data1[:, 1], 'o')
# plt.plot(new_data[:, 0], new_data[:, 1], 'o')
K = 2
m = np.zeros((K, 2))
random_initial = random.randint(0, n, size=K)
for j in range(K):
    m[j] = [new_data[random_initial[j], 0], new_data[random_initial[j], 1]]

dif = 1
itrt = 0
while dif != 0:
    itrt += 1
    cluster = np.ones(n)
    for i in range(n):
        dist = np.zeros(K)
        for j in range(K):
            dist[j] = distance(new_data[i], m[j])
        # now this data point has distances from all centroids
        dist_min = dist[0]
        pos = 0
        for j in range(K):
            if dist[j] < dist_min:
                dist_min = dist[j]
                pos = j
        # now pos is the closest centroid from this ith data
        cluster[i] = pos
        # now this data point is assigned to its nearest centroid
    # now all data points are assigned to their nearest centroids
    # re-calculate centroids
    nl = np.zeros(K)
    new_m = np.zeros((K, 2))
    for i in range(n):
        ac = int(cluster[i])  # assigned cluster
        nl[ac] += 1
        new_m[ac] += new_data[i]
    for j in range(K):
        new_m[j] = new_m[j] / nl[j]
    # calculate W sum of distances of all data points to their assigned centroids
    WC = 0
    for i in range(n):
        ac = int(cluster[i])  # assigned cluster
        WC += distance(new_data[i], new_m[ac])
    print(itrt)
    print('WC value is ', WC)
    # calculate difference between centroids
    dif = 0
    for k in range(K):
        dif += distance(m[k], new_m[k])
    # print(dif)
    m = new_m
# print(itrt)
# print(cluster)
# plot original data in cluster
for i in range(n):
    if cluster[i] == 0:
        plt.scatter(data1[i, 0], data1[i, 1], c='yellow')
    elif cluster[i] == 1:
        plt.scatter(data1[i, 0], data1[i, 1], c='cyan')
    else:
        plt.scatter(data1[i, 0], data1[i, 1], c='blue')  # in case
# plot data which are applied K-means
# for i in range(n):
#     if cluster[i] == 0:
#         plt.scatter(new_data[i, 0], new_data[i, 1], c='yellow')
#     elif cluster[i] == 1:
#         plt.scatter(new_data[i, 0], new_data[i, 1], c='cyan')
#     else:
#         plt.scatter(new_data[i, 0], new_data[i, 1], c='blue')  # in case
# plt.scatter(m[0, 0], m[0, 1], marker='x', c='black')
# plt.scatter(m[1, 0], m[1, 1], marker='x', c='green')

plt.show()
