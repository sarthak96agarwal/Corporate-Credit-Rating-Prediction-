import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.layers import Dense, Dropout
from keras import regularizers
from keras.models import Sequential, load_model
from collections import Counter
from keras.utils import to_categorical
from numpy import array
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
									 cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,classification_report
from scipy.interpolate import spline
from sklearn.decomposition import PCA
import pickle

tb = TensorBoard(log_dir='./Graph', histogram_freq=0, batch_size=32,
						   write_graph=True, write_grads=True, write_images=False,
						   embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

random_state = 23
checkpoint_exists = False

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

def w_categorical_crossentropy(y_true, y_pred, weights):
	nb_cl = len(weights)
	final_mask = K.zeros_like(y_pred[:, 0])
	y_pred_max = K.max(y_pred, axis=1)
	y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
	y_pred_max_mat = K.equal(y_pred, y_pred_max)
	for c_p, c_t in product(range(nb_cl), range(nb_cl)):
		final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
	return K.categorical_crossentropy(y_pred, y_true) * final_mask

def create_model(x,optimizer='adam', init='uniform',nf=98):
	print(f"Creating model with params -> optimizer={optimizer}, init={init}")
	model = Sequential()
	model.add(Dense(120, input_shape=(x.shape[1],), kernel_regularizer=regularizers.l2(0.02),
		kernel_initializer=init, activation='relu',))
	model.add(Dense(120, input_shape=(nf,),kernel_initializer=init, activation='relu',))
	model.add(Dense(64, kernel_initializer=init,activation='relu'))
	model.add(Dense(64, kernel_initializer=init,activation='relu'))
	model.add(Dense(32, kernel_initializer=init,activation='relu'))
	model.add(Dense(32, kernel_initializer=init,activation='relu'))
	model.add(Dense(32, kernel_initializer=init,activation='relu'))
	model.add(Dense(64, input_shape=(nf,),kernel_initializer=init, activation='relu',))
	model.add(Dense(5, kernel_initializer=init,activation='softmax'))
	model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
	return model

def get_model(x_train, y_train, optimizer='adam', init='normal',epochs=50,batch_size=15):
	model = create_model(optimizer=optimizer,init=init)
	print("Fitting model ... ")
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,callbacks=[tb, early_stop])
	return model

def checkpointed_fit(model, to_do=False, epochs=50, batch_size=64):
	callbacks_list = list()
	if to_do:
		filepath="best_weights.hdf5"
		checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		callbacks_list.append(checkpoint)
		checkpoint_exists = True
	model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
	return model

def grid_search(x_train, y_train):
	print("Performing GridSearch ...")
	model = KerasClassifier(build_fn=create_model, verbose=0)
	optimizers = ['rmsprop', 'adam']
	init = ['normal']
	epochs= [50, 100, 200, 300, 400]
	batches = [100,400]
	param_grid = dict(optimizer= optimizers, batch_size=batches, epochs=epochs, init=init)
	grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=5)
	grid_result = grid.fit(x_train, y_train)
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	return grid_result

def kfold(model, x_train, y_train):
	kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
	results = cross_val_score(model, x_train, y_train, cv=kfold,scoring="accuracy")
	print('kFold cross validation results = ', results.mean())

def make_y_orig(df):
	l = list()
	nc = 3
	for i in range(len(df['rating'])):
		if df['rating'].iloc[i][0]=='A':
			l.append(0)
		elif df['rating'].iloc[i][:2]=='BB':
			l.append(1)
		else:
			l.append(2)
	return l

def make_y(df):
	l = list()
	nc = 5
	pattern = ['AA', 'A', 'BBB', 'BB']
	# pattern = ['A','BBB+','BBB-','BB']
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

def redundant_code():
	x_train, x_dev, y_train, y_dev = train_test_split(x, y, stratify=y,test_size = 0.20, random_state=random_state)
	y_train_label = y_train
	y_dev_label = y_dev
	y_train = to_categorical(y_train, num_classes=nc)
	y_dev = to_categorical(y_dev, num_classes=nc)
	print("Stats after split")
	print("y_train")
	print(Counter(y_train_label).keys())
	print(Counter(y_train_label).values())
	print("y_dev")
	print(Counter(y_dev_label).keys())
	print(Counter(y_dev_label).values())
	grid_result = grid_search(x_train, y_train)

def epoch_accuracy_graph():	
	plt.figure()
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'validation'], loc='upper left')
	plt.show()
	print("Evaluating model ...")
	score = model.evaluate(x_dev, y_dev)
	training_score = model.evaluate(x_train, y_train)
	print('Score on dev set =',score[1])
	print('Training score = ', training_score[1])
	return (score[1], training_score[1])

def pca_(X, n_comp):
	pca = PCA(n_components=n_comp)
	pca.fit(X)
	print("Explained variance after PCA = ", sum(pca.explained_variance_ratio_))
	print("Transforming ...")
	X = pca.transform(X)
	return X

def run(nf,verbose=1,n_comp=0):
	print("Start ... !")
	print(f"Using {nf} features ... ")
	df = pd.read_csv('dataset.csv')
	thresh = 50
	df = clean_data(df, thresh)
	df = shuffle(df)
	print("Len of df after dropping nan-heavy rows = ", len(df))
	x = df.drop('rating',axis=1)
	# x = select_nf_features(x,nf)
	imputer = Imputer()
	x = imputer.fit_transform(x.values)
	if(n_comp>0):
		x = pca_(x,n_comp)
	scaler = StandardScaler()
	scaler.fit(x)
	x = scaler.transform(x)
	nc = 5 # num of classes
	y = make_y(df)
	ya = array(y)
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
	print("Training model ...")
	optimizer = 'adam'
	init='normal'
	epochs=400
	batch_size=100
	model = create_model(x,optimizer=optimizer,init=init,nf=nf)
	# print(model.summary())
	print("Fitting model with cross validation... ")
	kfold = StratifiedKFold(n_splits=7, shuffle=True, random_state=20)
	a = kfold.split(x, ya)
	cvscores = []
	for t in a:
		class_weights=dict()
		class_weights[0]=2
		class_weights[1]=1
		class_weights[2]=1
		class_weights[3]=1
		class_weights[4]=1
		model.fit(x[t[0]], to_categorical(ya[t[0]]), epochs=epochs, verbose=verbose, batch_size=batch_size,class_weight=class_weights)
		scores = model.evaluate(x[t[1]], to_categorical(ya[t[1]]), verbose=0)
		y_pred = model.predict_classes(np.array(x[t[1]]))
		y_true = ya[t[1]]
		target_names = ['0', '1', '2', '3', '4']
		# print(confusion_matrix(list(ya[t[1]]), list(preds)))
		print(classification_report(y_true,y_pred,target_names=target_names))
		print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
		cvscores.append(scores[1] * 100)
	print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
	return np.mean(cvscores)

# nf = 65
# run(nf,0)

# l = range(1,22,1)
# scores = list()
# for i in l:
# 	s = run(65,0,i)
# 	scores.append(s)

# print(scores)

# with open('nn_pca_list.pickle', 'wb') as handle:
# 	pickle.dump(l, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	print("pickle created")
# with open('nn_pca_scores.pickle', 'wb') as handle:
# 	pickle.dump(scores, handle, protocol=pickle.HIGHEST_PROTOCOL)
# 	print("pickle created")

# READ
with open('nn_pca_list.pickle', 'rb') as handle:
	l = pickle.load(handle)
with open('nn_pca_scores.pickle', 'rb') as handle:
	scores = pickle.load(handle)
print("MAX: ", max(scores))
lp = list()
scoresp = list()
for i in range(len(l)):
	if i%3 == 0:
		lp.append(l[i])
		scoresp.append(scores[i])

scores = scoresp
l = lp

# smoothen graph
xnew = np.linspace(l[0],l[-1],100)
power_smooth = spline(l,scores,xnew)

plt.figure()
plt.plot(xnew,power_smooth)
plt.xlabel('n_comp')
plt.ylabel('acc')
plt.title('No of PCA components vs. Accuracy for Neural Network')
plt.show()

# dt = sorted(enumerate(scores), key=operator.itemgetter(1))
# print(dt)
