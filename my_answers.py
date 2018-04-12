import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
	# containers for input/output pairs
	X = []
	y = []
	
	last_element_start = len(series) - window_size
	
	for ele in range(0, last_element_start, 1):
		current_series_last_element = ele+window_size
		X.append(series[ele : current_series_last_element])
		y.append(series[current_series_last_element])
	
	# reshape each 
	X = np.asarray(X)
	X.shape = (np.shape(X)[0:2])
	y = np.asarray(y)
	y.shape = (len(y),1)

	return X,y
	
# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
	model = Sequential()
	model.add(LSTM(5, input_shape=(window_size,1)))
	model.add(Dense(1, activation='linear'))
	return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
	punctuation = ['!', ',', '.', ':', ';', '?']
	stringlower = string.ascii_lowercase + ' '
	chartokeep = stringlower + ''.join(punctuation)
	
	for charele in text:
		if charele not in chartokeep:
			text = text.replace(charele, ' ')
	
	return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
	# containers for input/output pairs
	inputs = []
	outputs = []
	
	last_element_start = len(text) - window_size
	
	for ele in range(0, last_element_start, step_size):
		current_series_last_element = ele+window_size
		inputs.append(text[ele : current_series_last_element])
		outputs.append(text[current_series_last_element])
	
	return inputs,outputs
     
# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
	model = Sequential()
	model.add(LSTM(200, input_shape=(window_size, num_chars)))
	model.add(Dense(num_chars, activation='relu'))
	model.add(Dense(num_chars, activation='softmax'))
	return model