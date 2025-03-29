# -*- coding: utf-8 -*-
import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.decomposition import PCA
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
import numpy.random as random
import sys
import time
from pathlib import Path





len_feature = 1024 * 8  # 每行特征长度
num_patch = 8  # patch数量
len_patch = len_feature // num_patch  # patch特征长度

#path1 = '/data/jiangjiewei/dk/features/slit_lamp_by_nocropped/train/densenet121/processed_data.txt'
path1='/home/jiangjiewei/dingke_gpu/classifier_patch/features/features_by_patch_1024/densenet121/train/densenet121/processed_data.txt'
path3 = '/home/jiangjiewei/dingke_gpu/classifier_patch/features/features_by_patch_1024/densenet121/test/densenet121/processed_data.txt'
suffix = "_%s_20210722" % Path(path1).parent.name




data_train = np.genfromtxt(path1, dtype='float', delimiter=',')
x_train, y_train = np.split(data_train, (len_feature,), axis=1)

data_test = np.genfromtxt(path3, dtype='float', delimiter=',')
x_test, y_test = np.split(data_test, (len_feature,), axis=1)


#pca操作
len_train = len(x_train)
data = np.vstack((x_train, x_test))
pca = PCA(n_components=1024)
data = pca.fit_transform(data)
x_train, x_test = np.split(data, (len_train,), axis=0)


for i in range(len(y_test)):
    if y_test[i] == 0:
        with open('/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca/val/features_keratitis.txt', 'a') as f:
            for features_index in x_test[i]:
                f.write(str(features_index) + ' ')
            f.write('\n')
    elif y_test[i] == 1:
        with open('/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca/val/features_normal.txt', 'a') as f:
            for features_index in x_test[i]:
                f.write(str(features_index) + ' ')
            f.write('\n')
    elif y_test[i] == 2:
        with open('/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca/val/features_other.txt', 'a') as f:
            for features_index in x_test[i]:
                f.write(str(features_index) + ' ')
            f.write('\n')
#PCA训练集
for i in range(len(y_train)):
    if y_train[i] == 0:
        with open('/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca/train/features_keratitis.txt', 'a') as f:
            for features_index in x_train[i]:
                f.write(str(features_index) + ' ')
            f.write('\n')
    elif y_train[i] == 1:
        with open('/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca/train/features_normal.txt', 'a') as f:
            for features_index in x_train[i]:
                f.write(str(features_index) + ' ')
            f.write('\n')
    elif y_train[i] == 2:
        with open('/home/jiangjiewei/dingke_gpu/test_linear/features/feature_pca/train/features_other.txt', 'a') as f:
            for features_index in x_train[i]:
                f.write(str(features_index) + ' ')
            f.write('\n')




