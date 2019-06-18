import numpy as np
import os.path
from os.path import isdir, join
import librosa
import dtw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import norm
import random 

def k_means_clust(data,num_clust,num_iter,w=5):
	centroids=random.sample(data,num_clust)
	counter=0
	for n in range(num_iter):
		counter+=1
		print(counter)
		assignments={}
		#assign data points to clusters
		for ind,i in enumerate(data):
			min_dist=float('inf')
			closest_clust=None
			for c_ind,j in enumerate(centroids):
				dist, cost, acc_cost, path = dtw.dtw(i, j, dist=lambda x, y: norm(x - y, ord=1))
				if dist<min_dist:
					cur_dist= dist #DTWDistance(i,j,w)
					if cur_dist<min_dist:
						min_dist=cur_dist
						closest_clust=c_ind
			if closest_clust in assignments:
				assignments[closest_clust].append(ind)
			else:
				assignments[closest_clust]=[]
		
		#recalculate centroids of clusters
		for key in assignments:
			clust_sum = np.zeros(data[1].shape) 
			for k in assignments[key]:
				clust_sum=clust_sum+data[k]
				centroids[key]=[m/len(assignments[key]) for m in clust_sum]
	return centroids

files = []
filetype = "wav"
# Standard traversal with os.walk
dir = "C:/Users/vysha/Downloads/train/audio/cat"
for dirpath, dirnames, filenames in os.walk(dir):
    for filename in [f for f in filenames if f.endswith(filetype)]:
        files.append(os.path.join(dirpath, filename))

mfccs = []

for file in files:
	y, sr = librosa.load(file) #Load an audio file as a floating point time series.
	mfcc = librosa.feature.mfcc(y, sr) #get mel-frequency cepstral coefficients
	if mfcc.shape[1] != 44:
		shp = 44 - mfcc.shape[1]
		npad = ((0, 0), (0, shp))
		mfcc = np.pad(mfcc, pad_width=npad, mode='constant', constant_values=0)		
	mfccs.append(mfcc.T)

centroids=k_means_clust(mfccs,4,3,4)

for i in centroids:

    plt.plot(i)
plt.title("frequency clusters for command 'cat'");
plt.xlabel("time")
plt.ylabel("frequency")
plt.show()
