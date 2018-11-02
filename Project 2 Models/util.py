import pandas as pd
import matplotlib.pyplot as plt

def clean_data(df):
	print("Cleaning data ...")
	least = 50 # At least 60 out of 98 non nan values are required.
	print("Dropping rows with more than 30nans ...")
	df = df.dropna(axis=0,thresh=least)
	df = df.reset_index(drop=True)
	return df

def get_count(df):
	# pattern = ['AAA','AA', 'A', 'BBB', 'BB', 'B']
	# pattern = ['A','BBB+','BBB-','BB']
	pattern = ['AA', 'A', 'BBB', 'BB']
	ctr = [0,0,0,0,0]
	a = df['rating'].values.tolist()
	for i in a:
		flag=0
		for p in range(len(pattern)):
			if flag==0 and pattern[p] in i:
				ctr[p]=ctr[p]+1
				flag=1
			if flag==1:
				break
		if flag==0:
			ctr[-1]+=1
	return ctr

def make_y(df):
	l = list()
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

def nan_by_col(df):
	dff = df.isnull().sum()
	y = dff.index.values.tolist()
	x = dff.values.tolist()
	return (x,y)
	# plt.plot(y,x)
	# plt.xlabel('col name')
	# plt.ylabel('nan count')
	# plt.show()

df = pd.read_csv("dataset.csv")
df = clean_data(df)
ctr = get_count(df)
print(ctr)
print(sum(ctr))
for i in ctr:
	i = (i / sum(ctr)) * 100
	print(i)
# x,y = nan_by_col(df)
