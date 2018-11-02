from sklearn.neural_network import MLPClassifier
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
import pickle
import matplotlib.pyplot as plt

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
	# df = df.fillna(0)
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

def run(nf, thresh, n):
	# Remove nan data
	df = pd.read_csv('dataset.csv')
	df = clean_data(df, thresh=thresh)
	# X
	X = df.drop('rating',axis=1)
	print("Length of X = ", len(X))
	X = select_nf_features(X,nf)
	print("Columns = ", len(X.columns))
	imputer = Imputer()
	X = imputer.fit_transform(X.values)
	scaler = StandardScaler()
	scaler.fit(X)
	x = scaler.transform(X)
	y = make_y(df)

	x_train, x_dev, y_train, y_dev = train_test_split(x, y, stratify=y,test_size = 0.20, random_state=random_state)
	print('MLP Classifier')
	nn = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(120,64,32), random_state=1,max_iter=n)
	nn.fit(x_train, y_train)
	score =  nn.score(x_dev, y_dev)
	print("Accuracy = ", score)
	return score

# score = list()
# a = range(100,400,25)
# for i in range(100,400,25):
# 	s = run(56, 85,i)
# 	score.append(s)
# plt.figure()
# plt.plot(a,score)
# plt.xlabel('iter')
# plt.ylabel('acc')
# plt.show()
# print("MAX SCORE = ", max(score))
s = run(56,85,250)
