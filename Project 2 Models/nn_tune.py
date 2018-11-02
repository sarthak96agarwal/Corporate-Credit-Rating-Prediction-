from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from sklearn.metrics import roc_auc_score
import sys
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense
from keras.models import Sequential, load_model
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn import preprocessing
from collections import Counter

random_state = 23
checkpoint_exists = False

def shuffle(df, n=1, axis=0):
    print("Shuffling data ...")
    df = df.copy()
    for _ in range(n):
        df.apply(np.random.shuffle, axis=axis)
    return df

def clean_data(df):
    print("Cleaning data ...")
    least = 60 # At least 60 out of 98 non nan values are required.
    print("Dropping rows with more than 30nans ...")
    df = df.dropna(axis=0,thresh=least)
    df = df.reset_index(drop=True)
    return df

def create_model(optimizer='adam', init='uniform'):
    print(f"Creating model with params -> optimizer={optimizer}, init={init}")
    model = Sequential()
    model.add(Dense(120, input_shape=(98,), kernel_initializer=init, activation='relu'))
    model.add(Dense(64, kernel_initializer=init, activation='relu'))
    model.add(Dense(32, kernel_initializer=init, activation='relu'))
    model.add(Dense(10, kernel_initializer=init, activation='relu'))
    # model.add(Dense(3, kernel_initializer=init, activation='softmax'))
    model.add(Dense(5, kernel_initializer=init, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def get_model(x_train, y_train, optimizer='adam', init='normal',epochs=50,batch_size=15):
    model = create_model(optimizer=optimizer,init=init)
    print("Fitting model ... ")
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
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
    init = ['glorot_uniform', 'normal', 'uniform']
    epochs= [50, 100]
    batches = [10, 15, 30]
    param_grid = dict(optimizer= optimizers, batch_size=batches, epochs=epochs, init=init)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=5)
    grid_result = grid.fit(x_train, y_train)
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    return grid_result

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

print("Start ... !")
df = pd.read_csv('dataset.csv')
df = clean_data(df)
df = shuffle(df)

x = df.drop('rating',axis=1)
print("Len of X after dropping nan-heavy rows = ", len(x))
imputer = Imputer()
x = imputer.fit_transform(x.values)
scaler = StandardScaler()
scaler.fit(x)
x = scaler.transform(x)
nc = 0 # num of classes
y = make_y(df)

test = 1

x_train, x_dev, y_train, y_dev = train_test_split(x, y, stratify=y,test_size = 0.15, random_state=random_state)
if test:
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size = 0.15,random_state=random_state)
    y_test = to_categorical(y_test, num_classes=nc)
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

# grid_result = grid_search(x_train, y_train)
X = x_train
y = y_train
X_val = x_dev
y_val = y_dev

space = {'choice': hp.choice('num_layers',
                    [ {'layers':'two', },
                    {'layers':'three',
                    'units3': hp.uniform('units3', 64,256), 
                    }
                    ]),

            'units1': hp.uniform('units1', 64,256),
            'units2': hp.uniform('units2', 64,256),

            'batch_size' : hp.uniform('batch_size', 15,70),

            'nb_epochs' :  100,
            'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
            'activation': 'relu'
        }

def f_nn(params):   
    from keras.models import Sequential
    from keras.layers.core import Dense, Dropout, Activation
    from keras.optimizers import Adadelta, Adam, rmsprop

    print ('Params testing: ', params)
    model = Sequential()
    model.add(Dense(int(params['units1']), input_dim = 98, kernel_initializer="glorot_uniform",activation=params['activation']))
    model.add(Dense(int(params['units2']),kernel_initializer = "glorot_uniform", activation=params['activation']))
    if params['choice']['layers']== 'three':
        model.add(Dense(int(params['choice']['units3']),kernel_initializer = "glorot_uniform", activation=params['activation']))

    model.add(Dense(5,kernel_initializer = "glorot_uniform", activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
    model.fit(X, y, nb_epoch=int(params['nb_epochs']), batch_size=int(params['batch_size']), verbose = 0)
    score = model.evaluate(x_dev, y_dev)
    training_score = model.evaluate(x_train, y_train)
    test_score = model.evaluate(x_test,y_test)
    print('Score on dev set =',score[1]) 
    print('Training score = ', training_score[1])
    print('Score on test set = ', test_score[1])
    # pred_auc = model.predict_proba(X_val, batch_size = 128, verbose = 0)
    # acc = roc_auc_score(y_val, pred_auc)
    # print('AUC:', acc)
    # sys.stdout.flush() 
    return {'loss': -score[1], 'status': STATUS_OK}

trials = Trials()
best = fmin(f_nn, space, algo=tpe.suggest, max_evals=50, trials=trials)
print('best')
print(best)
