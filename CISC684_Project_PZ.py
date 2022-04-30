import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from PIL import Image
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


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


path0 = r'D:\Data Science Courses\CISC684\Project'
path1 = r'D:\Data Science Courses\CISC684\Project\images'
os.chdir(path1)
new_size = (24, 24)
category_name = []
category_index = []
image_name = []
index = 0
xdata = []
# convert image files into data array
# n*d where d is size of reshaped image for example 24*24 = 576
for directory in os.listdir(path1):
    index += 1
    path2 = path1 + '\\' + directory
    os.chdir(path2)
    for file in glob.glob('*.jpg'):
        # y label
        category_name.append(directory)
        category_index.append(index)
        image_name.append(file)
        # x data
        img = Image.open(file)
        img_resized = img.resize(new_size)
        img_grayscale = img_resized.convert('L')
        # img_rawdata = np.array(img.getdata())   #vectorized
        img_graydata = np.array(img_grayscale.getdata())  # vectorized
        xdata.append(img_graydata)
X = np.array(xdata)
# x = np.transpose(xdata) # x is d*n
df = pd.DataFrame({'folder name': category_name,
                   'category': category_index,
                   'image name': image_name})
os.chdir(path0)
if os.path.isfile('label.csv'):
    # print('label file exists')
    os.remove('label.csv')
    df.to_csv('label.csv')
else:
    df.to_csv('label.csv')
# y data n*1 values are ranging from 1-120
y = np.array(category_index)
n = len(y)
cat = []
index_count = 0
for k in range(n):
    if category_index[k] != index_count:
        cat.append(category_name[k])
        index_count += 1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# To check coding accuracy for specific folder - one species
# path10 = r'D:\Data Science Courses\CISC684\Project\images\n02085782-Japanese_spaniel'
# os.chdir(path10)
# count = 0
# xdata_0 = []
# for file in glob.glob('*.jpg'):
#     count += 1
#     img = Image.open(file)
#     img_resized = img.resize(new_size)
#     img_grayscale = img_resized.convert('L')
#     img_rawdata = np.array(img.getdata())
#     img_graydata = np.array(img_grayscale.getdata())
#     xdata_0.append(img_graydata)
# print(img_rawdata.shape)
# print(img_graydata.shape)
# plt.imshow(img)
# plt.imshow(img_resized)
# plt.imshow(img_grayscale, cmap='gray')

# PCA to reduce dimension
# x_0 = np.transpose(xdata)
# dimension = new_size[0] * new_size[1]
# s_matrix = covariance_matrix(x_0)
# U, lambdas, UT = np.linalg.svd(s_matrix, full_matrices=True)
# sum_lambda = sum(lambdas)
# ratio = 0
# k = 0
# while ratio < 0.90:
#     ratio = sum(lambdas[:k]) / sum_lambda
#     k += 1
# print(k)
# ratio = np.zeros(dimension)
# for i in range(dimension):
#     ratio[i] = sum(lambdas[:i]) / sum_lambda
# x_scale = np.linspace(0, dimension, dimension)
# # plt.plot(x_scale, lambdas)
# plt.plot(x_scale, ratio, label='ratio')
# plot eigen-vectors
# figure, axis = plt.subplots(4, 5)
# x_bar = np.mean(x_0, axis=1)
# x_bar = x_bar.reshape(new_size)
# axis[0, 0].imshow(x_bar)
# axis[0, 0].set_title('Eigenimage ' + str(0))
# for z in range(19):
#     image = U[:, z].reshape(new_size)
#     r = (z + 1) // 5
#     c = (z + 1) % 5
#     axis[r, c].imshow(image)
#     axis[r, c].set_title('Eigenimage ' + str(z + 1))

# SVM
h = 0.02
C = 1.0  # SVM regularization parameter
svc = SVC(kernel='poly', C=C, degree=3).fit(X_train, y_train)
# svc = SVC(kernel='poly', C=C, degree=10).fit(X_train, y_train)
# svc = SVC(kernel='rbf', C=C, gamma=0.1).fit(X_train, y_train)
# svc = SVC(kernel='rbf', C=C, gamma=10).fit(X_train, y_train)
y_train_pred = svc.predict(X_train)
length_train = len(X_train)
accuracy_train = sum(y_train_pred == y_train) / length_train
print('Train set accuracy is :', accuracy_train)
y_pred = svc.predict(X_test)
length_test = len(X_test)
accuracy_test = sum(y_pred == y_test) / length_test
print('Total images in test set is', length_test)
print('Total wrong predict images are', sum(y_pred != y_test))
print('Test set accuracy is :', accuracy_test)
# for i in range(length_test):
#     if y_pred[i] != y_test[i]:
#         test_image = np.reshape(X_test[i], new_size)
#         plt.imshow(test_image, cmap='gray')
#         print('real image is' + cat[y_test[i]-1])
#         print('predicted image is' + cat[y_pred[i]-1])
#         break
plt.show()
