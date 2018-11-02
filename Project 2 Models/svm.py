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
from scipy.interpolate import spline
import pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
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

def pca_(X, n_comp):
	pca = PCA(n_components=n_comp)
	pca.fit(X)
	print("Explained variance after PCA = ", sum(pca.explained_variance_ratio_))
	print("Transforming ...")
	X = pca.transform(X)
	return X

def run(nf, thresh, n_comp=0):
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
	# if(n_comp>0):
	# 	X = pca_(X,n_comp)
	scaler = StandardScaler()
	scaler.fit(X)
	x = scaler.transform(X)

	y = make_y(df)

	# Split data into training and test set
	x_train, x_dev, y_train, y_dev = train_test_split(x, y, stratify=y,test_size = 0.20, random_state=random_state)

	print('SVC')
	# C = 1000000000
	C = 100000000
	g = 0.0000001
	clf = svm.SVC(kernel='sigmoid', C=C, verbose=0, gamma=g, coef0=0.001)
	print(f"C = {C} and gamma = {g}")
	clf.fit(x_train, y_train)
	score = clf.score(x_dev, y_dev)
	y_pred = clf.predict(x_dev)
	target_names = ['0', '1', '2', '3', '4']
	y_true = y_dev
	print(confusion_matrix(y_true, y_pred))
	print(classification_report(y_true, y_pred, target_names=target_names))
	print("Accuracy = ", score)
	return score

# score = list()
# a = range(1,30,3)
# for i in a:
# 	s = run(70, 80, i)
# 	score.append(s)

# with open('svm_features_list.pickle', 'wb') as handle:
#     pickle.dump(a, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("pickle created")
# with open('svm_features_scores.pickle', 'wb') as handle:
#     pickle.dump(score, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("pickle created")

# with open('svm_features_list.pickle', 'rb') as handle:
# 	a = pickle.load(handle)
# with open('svm_features_scores.pickle', 'rb') as handle:
# 	score = pickle.load(handle)

# # smoothen graph
# ap = list()
# scorep = list()
# for i in range(len(a)):
# 	if i%2 == 0:
# 		ap.append(a[i])
# 		scorep.append(score[i])

# score = scorep
# a = ap
# xnew = np.linspace(a[0],a[-1],100)
# power_smooth = spline(a,score,xnew)

# plt.figure()
# plt.plot(xnew, power_smooth)
# plt.title('No of PCA components vs. Accuracy for SVM')
# plt.xlabel('n_comp')
# plt.ylabel('acc')
# plt.show()

s = run(70,80)
