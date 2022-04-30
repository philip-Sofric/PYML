import scipy.io
import os

# n02085620-Chihuahua/n02085620_5927.jpg
def parse_filename(filename):
    sec_1 = ''
    sec_2 = ''
    for character in filename:
        if


path1 = r'D:\Data Science Courses\CISC684\Project\lists'
path2 = r'D:\Data Science Courses\CISC684\Project\images'
os.chdir(path1)
train_list_set = scipy.io.loadmat('train_list.mat')
train_filenames = train_list_set['file_list']
train_labels = train_list_set['labels']
# print(train_label.shape)
test_list_set = scipy.io.loadmat('test_list.mat')
test_filenames = test_list_set['file_list']
test_labels = test_list_set['labels']
for filename in train_filenames:
