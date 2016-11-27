import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

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