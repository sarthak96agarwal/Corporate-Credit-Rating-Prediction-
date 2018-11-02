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
	df = df.reset_index(drop=True)
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
	kfold = StratifiedKFold(n_splits=6, shuffle=True, random_state=20)
	a = kfold.split(x, ya)
	cvscores = []
	for t in a:
		n_estimators=32
		max_features=5
		clf = RandomForestClassifier(n_estimators=300, criterion='entropy', max_features=50)
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

run(77,85)
# run(77,85) ERF

# l = range(5,50,3)
# scores = list()
# for i in l:
# 	scores.append(run(i,85))
# plt.figure()
# xnew = np.linspace(l[0],l[-1],100)
# power_smooth = spline(l,scores,xnew)
# plt.plot(xnew, power_smooth)
# # plt.plot(l,scores)
# plt.title('No of PCA components vs. Accuracy for RandomForestClassifier')
# plt.xlabel('n_comp')
# plt.ylabel('acc')
# plt.show()
# print("MAX: ", max(scores))
# dt = sorted(enumerate(scores), key=operator.itemgetter(1))
# print(dt)
