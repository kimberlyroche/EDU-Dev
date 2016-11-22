import numpy as np
import pandas
# import sqlite3
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
# from sklearn.pipeline import Pipeline


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(4, input_dim=4, init='normal', activation='relu'))
	model.add(Dense(3, init='normal', activation='sigmoid'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

binary = False

seed = 7
np.random.seed(seed)

if(binary):
	# BINARY CLASSIFIER
	dataset = np.loadtxt("pima.csv", delimiter=",")
	X = dataset[:,0:8]
	Y = dataset[:,8]

	model = Sequential()
	model.add(Dense(12, input_dim=8, init='uniform', activation='relu')) # hidden layer 1
	model.add(Dense(8, init='uniform', activation='relu')) # hidden layer 2
	model.add(Dense(1, init='uniform', activation='sigmoid')) # output

	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	model.fit(X, Y, nb_epoch=150, batch_size=10, verbose=0)

	scores = model.evaluate(X, Y)
	print "loss = %.4f" % scores[0]
	print "accuracy = %.0f%%" % (scores[1]*100)

	predictions = model.predict(X[0,:].reshape((1, 8)))
	print predictions
	print [round(x) for x in predictions]
else:
	dataframe = pandas.read_csv("iris.csv", header=None)
	dataset = dataframe.values
	X = dataset[:,0:4].astype(float)
	Y = dataset[:,4]

	encoder = LabelEncoder()
	encoder.fit(Y)
	encoded_Y = encoder.transform(Y)
	dummy_y = np_utils.to_categorical(encoded_Y)
	X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.2, random_state=seed)
	# X_train, X_test, Y_train, Y_test = train_test_split(X, dummy_y, test_size=0.2)

	estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=200, batch_size=5, verbose=0)

	# use kfold (10-fold) cross-validation to "describes the evaluation of the 10 
	#		constructed models for each of the splits of the dataset"
	# kfold = KFold(n_splits=10, shuffle=True, random_state=seed) # 10-fold cross-validation
	# results = cross_val_score(estimator, X, dummy_y, cv=kfold)
	# print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))

	estimator.fit(X_train, Y_train)
	print estimator.score(X_test, Y_test)
	# predictions = estimator.predict(X_test)
	# print(predictions)
	# print(encoder.inverse_transform(predictions))



# NEXT:
#	1. Can we forego the KerasClassifier class and try a by-hand (as binary above) still getting good scoring? (use iris.csv)
# 	2. Binning no. STEM courses output variable by 0.2 percentiles
#	3. Merge transform_data.py with this file
#	4. Encoding multiple output variables as individual classes:
# 		http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html



