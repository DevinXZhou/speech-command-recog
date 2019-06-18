import numpy as np
import os.path
from os.path import isdir, join
import librosa
import dtw
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from numpy.linalg import norm

y, sr = librosa.load("C:/Users/vysha/Downloads/train/audio/cat/2c7c33e8_nohash_1.wav")
mfcc1 = librosa.feature.mfcc(y, sr)
y, sr = librosa.load("C:/Users/vysha/Downloads/train/audio/cat/1ffd513b_nohash_1.wav")
mfcc2 = librosa.feature.mfcc(y, sr)

dist, cost, acc_cost, path = dtw.dtw(mfcc1.T, mfcc2.T, dist=lambda x, y: norm(x - y, ord=1))
print ('Normalized distance between the two sounds:', dist)

plt.imshow(cost.T, origin='lower', cmap=cm.gray, interpolation='nearest')
plt.plot(path[0], path[1], 'w')
plt.show()
plt.xlim((-0.5, cost.shape[0]-0.5))
plt.ylim((-0.5, cost.shape[1]-0.5))