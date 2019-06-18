'''
Created on Oct 21, 2017

@author: devin

'''

from dsp.pca import PCA
from dsp.utils import pca_analysis_of_spectra, get_labels_path
from dsp.utils import get_corpus, Timer, \
    extract_features_from_corpus
from dsp.features import get_features


from keras.layers import Dense
import numpy as np



############## ADDED ##############
import os
import wave
import contextlib
from numpy import inf

from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.model_selection import StratifiedKFold
from keras.utils import np_utils
from keras.models import model_from_json

from sklearn.preprocessing import MinMaxScaler
import csv

from dsp.audioframes import AudioFrames
from dsp.rmsstream import RMSStream
############## ----- ##############

def fileFiltSave(files, nameX, nameY):
	adv_ms = 10
	len_ms = 20
	offset_s = 0.35

	filtFiles = []
	silenceFiles = []
	for file in files:
		with contextlib.closing(wave.open(file,'r')) as f:
		    frames = f.getnframes()
		    rate = f.getframerate()
		    duration = frames / float(rate)

		    if duration > 0.8 and duration < 1.2:
		    	filtFiles.append(file)

	files = []
	for f in filtFiles:
		framestream = AudioFrames(f, adv_ms, len_ms)
		rms = RMSStream(framestream)
		data = []
		for value in rms:
			data.append(value)
		std = np.std(data)
		if std < 6: #################################################Customize threshold
			silenceFiles.append(f)
		else:
			files.append(f)

	np.savetxt(nameX, files, delimiter=" ", fmt="%s")
	if nameY != 'train':
		np.savetxt(nameY, silenceFiles, delimiter=" ", fmt="%s")
	return [files, silenceFiles]

   
   
def main():
	adv_ms = 10
	len_ms = 20
	offset_s = 0.35
	trdir = os.getcwd()+"/train"
	tedir = os.getcwd()+"/test"
	components = 28

	
	# get training files
	try:
		trainFiles = np.genfromtxt('trainFileList.txt',dtype='str')
		[train_Y, trainFiles] = get_labels_path(trainFiles)
		print('Train Files List Loaded')
	except IOError:
		trainFiles = sorted(get_corpus(trdir))
		[trainFiles, trSilences] = fileFiltSave(trainFiles,'trainFileList.txt', 'train')
		del trSilences
		[train_Y, trainFiles] = get_labels_path(trainFiles)
		print('Train Files List Saved')

	# get testing files
	try:
		testFiles = np.genfromtxt('testFileList.txt',dtype='str')
		testSilences = np.genfromtxt('teSilenceList.txt',dtype='str')
		print('Test Files List Loaded')
	except IOError:
		testFiles = np.array(sorted(get_corpus(tedir)))
		[testFiles, testSilences] = fileFiltSave(testFiles,'testFileList.txt', 'teSilenceList.txt')
		print('Test Files List Saved')



	try:
	#load train data
		train_X = np.fromfile('trainData.dat', dtype=float)
		samplesN = trainFiles.shape[0]
		data_dim = components
		timesteps = int(train_X.shape[0]/samplesN/data_dim)
		

		train_X = np.reshape(train_X, (samplesN, timesteps, data_dim))
		print('Train Data Features Loaded')

		# load test data
		test_X = np.fromfile('testData.dat', dtype=float)
		test_samplesN = testFiles.shape[0]
		test_X = np.reshape(test_X, (test_samplesN, timesteps, data_dim))
		print('Test Data Features Loaded')

	except IOError:
		timer = Timer()
		pca = pca_analysis_of_spectra(trainFiles, adv_ms, len_ms, offset_s)
		print("PCA feature generation and analysis time {}, feature extraction..."
			.format(timer.elapsed()))

		timer.reset()
		# Read features - each row is a feature vector
		
		train_X = extract_features_from_corpus(
			trainFiles, adv_ms, len_ms, offset_s, pca, components)

		print("Time to generate features {}".format(timer.elapsed()))
		timer.reset()
		[samplesN, data_dim] = train_X.shape
		timesteps = 1;

		train_X.tofile('trainData.dat')
		print('Train Data Features Saved')
		train_X = train_X.flatten()
		train_X = np.reshape(train_X, (samplesN, timesteps, data_dim))

		# Read features - each row is a feature vector
		test_X = extract_features_from_corpus(
			testFiles, adv_ms, len_ms, offset_s, pca, components)

		[test_samplesN, data_dim] = test_X.shape

		test_X.tofile('testData.dat')
		print('Test Data Features Saved')
		test_X = test_X.flatten()
		test_X = np.reshape(test_X, (test_samplesN, timesteps, data_dim))



	num_classes = len(set(train_Y))


	try:
		json_file = open('model.json', 'r')
		loaded_model_json = json_file.read()
		json_file.close()
		model = model_from_json(loaded_model_json)
		# load weights into new model
		model.load_weights("LSTMgoogle.h5")
		print("Loaded model from disk")

		model.compile(loss='categorical_crossentropy',
		              optimizer='rmsprop',
		              metrics=['accuracy'])
	except IOError:

		model = Sequential()
		model.add(LSTM(256, return_sequences=True,
		               input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
		model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
		model.add(LSTM(256, return_sequences=True))  # returns a sequence of vectors of dimension 32
		model.add(LSTM(256))  # return a single vector of dimension 32
		model.add(Dense(num_classes, activation='softmax'))

		model.compile(loss='categorical_crossentropy',
		              optimizer='rmsprop',
		              metrics=['accuracy'])

		kfold = StratifiedKFold(2, shuffle=True)

		for (train_idx, test_idx) in kfold.split(train_X, train_Y):
			onehotlabels = np_utils.to_categorical(train_Y)
			model.fit(train_X[train_idx], onehotlabels[train_idx], 
				batch_size=256, epochs=100, validation_data=(train_X[test_idx], onehotlabels[test_idx]))
		model_json = model.to_json()
		with open("model.json", "w") as json_file:
		    json_file.write(model_json)
		model.save_weights("LSTMgoogle.h5")
		print('Model has been saved')



	classmap = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']

	filenames = []
	for f in testFiles:
		arr = f.split('/')
		filenames.append(arr[-1])


	result = model.predict(test_X)
	[n, bits] = result.shape
	print('obtained total predictions of ')
	print(n)
	dictionary = {}


	for f in testSilences:
		arr = f.split('/')
		name = arr[-1]
		dictionary[name] = 'silence'

	for i in range(0, n):
		classIdx = np.argmax(result[i,:])
		confidence = np.max(result[i,:])
		if classIdx < 10 and confidence > 0.95:
			dictionary[filenames[i]] = classmap[classIdx]
		else:
			dictionary[filenames[i]] = 'unknown'

	with open('submission.csv', 'w') as f:
		fieldnames = ['fname', 'label']
		writer = csv.DictWriter(f, fieldnames=fieldnames)
		writer.writeheader()
		data = [dict(zip(fieldnames, [k, v])) for k, v in dictionary.items()]
		writer.writerows(data)


if __name__ == '__main__':

	main()


