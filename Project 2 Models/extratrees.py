from sklearn import svm
import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
									 cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.neighbors import DistanceMetric
import pickle
import operator
from numpy import array
import matplotlib.pyplot as plt
from scipy.interpolate import spline
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
	print('dropped')
	df = df.reset_index(drop=True)
	print('returning')
	return df

def make_y(df):
	l = list()
	nc = 5
	for i in range(len(df['rating'])):
		if 'AA' in df['rating'].iloc[i]:
			l.append(0)
		elif 'A' in df['rating'].iloc[i]:
			l.append(1)
		elif 'BBB' in df['rating'].iloc[i]:
			l.append(2)
		elif 'BB' in df['rating'].iloc[i]:
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

def run(nf, thresh, n_comp=0):
	df = pd.read_csv('dataset.csv')
	df = clean_data(df, thresh=thresh)
	X = df.drop('rating',axis=1)
	# X = select_nf_features(X,nf)
	print("Columns = ", len(X.columns))
	imputer = Imputer()
	X = imputer.fit_transform(X.values)
	if(n_comp>0):
		X = pca_(X,n_comp)
	scaler = StandardScaler()
	scaler.fit(X)
	x = scaler.transform(X)
	y = make_y(df)
	ya = array(y)
	kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=20)
	a = kfold.split(x, ya)
	cvscores = []
	for t in a:
		# clf = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features=50)
		clf = ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_features=10, max_depth=50)
		clf.fit(x[t[0]], ya[t[0]])
		y_pred = clf.predict(x[t[1]])
		scores = clf.score(x[t[1]], ya[t[1]])
		target_names = ['0', '1', '2', '3', '4']
		y_true = ya[t[1]]
		print(confusion_matrix(list(ya[t[1]]), list(y_pred)))
		print(classification_report(y_true, y_pred, target_names=target_names))
		print("acc: %.2f%%" % (scores*100))
		cvscores.append(scores * 100)
	mean_score = np.mean(cvscores)
	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	return mean_score

def grid_search(x_train, y_train):
	print("Performing GridSearch ...")
	model = ExtraTreesClassifier(n_estimators=300, criterion='entropy', max_features=10, max_depth=5)
	optimizers = ['rmsprop', 'adam']
	init = ['normal']
	epochs= [50, 100, 200, 300, 400]
	batches = [100,400]
	param_grid = dict(optimizer= optimizers, batch_size=batches, epochs=epochs, init=init)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=5)
	grid_result = grid.fit(x_train, y_train)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	return grid_result

# MAX EFF
run(60,85)

# df = pd.read_csv('dataset.csv')
# df = clean_data(df, thresh=85)
# X = df.drop('rating',axis=1)
# imputer = Imputer()
# X = imputer.fit_transform(X.values)
# print('imputed')
# scaler = StandardScaler()
# scaler.fit(X)
# x = scaler.transform(X)
# y = make_y(df)
# ya = array(y)

# rfc = RandomForestClassifier() 
 
# # Use a grid over parameters of interest
# param_grid = { 
#            # "n_estimators" : [300],
#            "max_features" : [10, 70, 80, 90],
#            "max_depth" : [5, 10, 20, 30],
#            # "min_samples_leaf" : [1, 2, 6,10]
#            }
 
# print('searching now')
# CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=3, verbose=1)
# CV_rfc.fit(x,y)
# print(CV_rfc.best_params_)