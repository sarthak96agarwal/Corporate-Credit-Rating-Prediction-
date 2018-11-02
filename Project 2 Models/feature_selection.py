import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import operator
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif

def shuffle(df, n=1, axis=0):
	df = df.copy()
	for _ in range(n):
		df.apply(np.random.shuffle, axis=axis)
	return df

# X
df = pd.read_csv('dataset.csv')
X = df.drop('rating',axis=1)
imputer = Imputer()
X = imputer.fit_transform(X.values)

# Y
l = list()
for i in range(len(df['rating'])):
	if df['rating'][i][0]=='A':
		l.append(1)
	elif df['rating'][i][:2]=='BB':
		l.append(2)
	else:
		l.append(3)
Y = l

X_indices = np.arange(X.shape[-1])
selector = SelectPercentile(f_classif, percentile=10)
selector.fit(X,Y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
# plt.bar(X_indices - .45, scores, width=.2,label=r'Univariate score ($-Log(p_{value})$)',
# 	color='darkorange',
# 	edgecolor='black')
data = sorted(enumerate(scores), key=operator.itemgetter(1))
# with open('variance_features.pickle', 'wb') as handle:
#     pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
#     print("pickle created")
# print(data)
nf = 98
k = list()
print(f"Top {nf} features by percentile are -")
for i in range(nf):
	k.append(df.columns[data[-i-1][0]])
	print(f"{df.columns[data[-i-1][0]]} -> {data[-i-1][1]}")

# k.append('')
# k.append('')
# # table code
# import numpy as np
# import matplotlib.pylab as pl
# col = 10
# row = 10
# data = np.array(k).reshape(row, col);
# pl.figure()
# tb = pl.table(cellText=data, loc=(0,0), cellLoc='center')
# tc = tb.properties()['child_artists']
# for cell in tc: 
#     cell.set_height(1/row)
#     cell.set_width(1/col)
# ax = pl.gca()
# ax.set_xticks([])
# ax.set_yticks([])
# pl.show()

plt.show()
