#!/usr/bin/python

from __future__ import division

import numpy as np
import xgboost as xgb
from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
									 cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
import pickle
from scipy.interpolate import spline
import matplotlib.pyplot as plt
from numpy import array
from sklearn.metrics import confusion_matrix,classification_report

random_state = 42

def shuffle(df, n=1, axis=0):
	print("Shuffling data ...")
	df = df.copy()
	for _ in range(n):
		df.apply(np.random.shuffle, axis=axis)
	return df

def clean_data(df, thresh=60):
	print("Cleaning data ...")
	print(f"Dropping rows with more than {98-thresh} nans ...")
	df = df.dropna(axis=0,thresh=thresh)
	df = df.reset_index(drop=True)
	return df

def make_y(df):
	l = list()
	nc = 5
	pattern = ['AA', 'A', 'BBB', 'BB']
	for i in range(len(df['rating'])):
		if pattern[0] in df['rating'].iloc[i]:
			l.append(0)
		elif pattern[1] in df['rating'].iloc[i]:
			l.append(1)
		elif pattern[2] in df['rating'].iloc[i]:
			l.append(2)
		elif pattern[3] in df['rating'].iloc[i]:
			l.append(3)
		else:
			l.append(4)
	return l

def select_nf_features(df, nf):
	with open('variance_features.pickle', 'rb') as handle:
		data = pickle.load(handle)
	l = list()
	for i in data:
		l.append(i[0])
	l = l[-nf:] # Top nf features. >0.5 var
	cols = df.columns[l]
	df = df[cols]
	return df

def pca_(X, n_comp):
	pca = PCA(n_components=n_comp)
	pca.fit(X)
	print("Explained variance after PCA = ", sum(pca.explained_variance_ratio_))
	print("Transforming ...")
	X = pca.transform(X)
	return X

df = pd.read_csv('dataset.csv')
df = clean_data(df, thresh=thresh)

data = array(df)
sz = data.shape

train = data[:int(sz[0] * 0.7), :]
test = data[int(sz[0] * 0.7):, :]

train_X = train[:, :97]
train_Y = train[:, 98]

test_X = test[:, :97]
test_Y = test[:, 98]

xg_train = xgb.DMatrix(train_X, label=train_Y)
xg_test = xgb.DMatrix(test_X, label=test_Y)
# setup parameters for xgboost
param = {}
# use softmax multi-class classification
param['objective'] = 'multi:softmax'
# scale weight of positive examples
param['eta'] = 0.1
param['max_depth'] = 6
param['silent'] = 1
param['nthread'] = 4
param['num_class'] = 5

watchlist = [(xg_train, 'train'), (xg_test, 'test')]
num_round = 5
bst = xgb.train(param, xg_train, num_round, watchlist)
# get prediction
pred = bst.predict(xg_test)
error_rate = np.sum(pred != test_Y) / test_Y.shape[0]
print('Test error using softmax = {}'.format(error_rate))

# do the same thing again, but output probabilities
param['objective'] = 'multi:softprob'
bst = xgb.train(param, xg_train, num_round, watchlist)
# Note: this convention has been changed since xgboost-unity
# get prediction, this is in 1D array, need reshape to (ndata, nclass)
pred_prob = bst.predict(xg_test).reshape(test_Y.shape[0], 6)
pred_label = np.argmax(pred_prob, axis=1)
error_rate = np.sum(pred_label != test_Y) / test_Y.shape[0]
print('Test error using softprob = {}'.format(error_rate))
