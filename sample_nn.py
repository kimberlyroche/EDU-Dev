import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets
from sklearn import linear_model
import matplotlib

# activation_v = tanh(input_v * weight1_v + bias1_v)
# output_v = softmax(activation_v * weight2_v + bias2_v)

# size weight1_v = input size x hidden layer size
# size bias1_v = hidden layer size
# size weight2_v = hidden layer size x output size
# size bias2_v = output size

# these ^^^ are what we're assigning with training

# error is measured with the cross-entropy function:
#	loss = (-1/training_set) sum(training sets) { sum(outputs) { correct_output * log(produced output) } }

def scale_to_range(OldMin, OldMax, NewMin, NewMax, OldValue):
	OldRange = (OldMax - OldMin)
	NewRange = (NewMax - NewMin)
	NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
	return NewValue

np.random.seed(0)
# make random data in two interleaved half circles
x, y = datasets.make_moons(n_samples=200, noise=0.20)

nn_input_dim = 2 # x and y coords
nn_output_dim = 2 # output classes in 01 or 10 format

withhold_percent = 0.2 # % points to without as validation set
full_set_sz = len(x)
validation_set_sz = int(np.floor(full_set_sz * withhold_percent))
training_set_sz = int(np.ceil(full_set_sz * (1 - withhold_percent)))

validation_x = x.copy()
validation_x = validation_x[-validation_set_sz:,:]
validation_y = y.copy()
validation_y = validation_y[-validation_set_sz:]

x = x[0:training_set_sz,:]
y = y[0:training_set_sz]

epsilon = 0.01 # fixed gradient descent learning rate
reg_lambda = 0.01 # regularization strength; reduce overfitting

formatted_y = np.ndarray([training_set_sz, nn_output_dim])
for i in range(training_set_sz):
	if y[i] == 0:
		formatted_y.itemset((i, 0), 1)
		formatted_y.itemset((i, 1), 0)
	else:
		formatted_y.itemset((i, 0), 0)
		formatted_y.itemset((i, 1), 1)
formatted_valid_y = np.ndarray([validation_set_sz, nn_output_dim])
for i in range(validation_set_sz):
	if y[i] == 0:
		formatted_valid_y.itemset((i, 0), 1)
		formatted_valid_y.itemset((i, 1), 0)
	else:
		formatted_valid_y.itemset((i, 0), 0)
		formatted_valid_y.itemset((i, 1), 1)

def plot_decision_boundary(pred_func):
	# set your plot boundaries comfortably past the min and max
	x_min, x_max = validation_x[:,0].min() - 0.5, validation_x[:,0].max() + 0.5
	y_min, y_max = validation_x[:,1].min() - 0.5, validation_x[:,1].max() + 0.5
	# grid increment amount
	h = 0.01
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
	# c_ and r_ collapse arrays into a single flat list quickly
	# pass every coordinate np.c_[X, Y] to the predict function to apply the trained NN
	z = pred_func(np.c_[xx.ravel(), yy.ravel()])
	z = z.reshape(xx.shape)
	plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
	plt.scatter(validation_x[:,0], validation_x[:,1], c=validation_y, cmap=plt.cm.Spectral)
	plt.grid(True)
	plt.show()

# predict the output: 0 or 1
def predict(model, x):
	w1, b1, w2, b2 = model['w1'], model['b1'], model['w2'], model['b2']
	z1 = x.dot(w1) + b1
	a1 = np.tanh(z1)
	z2 = a1.dot(w2) + b2
	exp_scores = np.exp(z2)
	softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	# return max of two columns (0, 1)
	return np.argmax(softmax, axis=1)

def build_model(nn_hdim=3, epochs=1000, print_loss=False, batch_sz=40):
	w1 = np.random.randn(nn_input_dim, nn_hdim)
	b1 = np.zeros((1, nn_hdim))
	w2 = np.random.randn(nn_hdim, nn_output_dim)
	b2 = np.zeros((1, nn_output_dim))

	model = {}

	for i in range(epochs):
		shuffled_indices = range(training_set_sz)
		np.random.shuffle(shuffled_indices)
		lower = 0
		upper = lower + batch_sz
		while lower < training_set_sz:
			# print "Batch range %d to %d" % (lower, upper)
			batch_x = [x[shuffled_indices[lower]]]
			batch_y = [formatted_y[shuffled_indices[lower]]]
			for j in range(lower+1, upper):
				batch_x = np.concatenate((batch_x,[x[shuffled_indices[j]]]), axis=0)
				batch_y = np.concatenate((batch_y,[formatted_y[shuffled_indices[j]]]), axis=0)

			# forward propogation
			# z1 = x.dot(w1) + b1
			z1 = batch_x.dot(w1) + b1
			a1 = np.tanh(z1)
			z2 = a1.dot(w2) + b2
			exp_scores = np.exp(z2)
			softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

			# backpropogation
			# delta3 = softmax - formatted_y
			delta3 = softmax - batch_y
			dw2 = (a1.T).dot(delta3)
			db2 = np.sum(delta3, axis=0, keepdims=True) # sum on rows, don't flatten
			delta2 = (-np.square(np.tanh(z1)) + 1) * (delta3.dot(w2.T))
			# dw1 = (x.T).dot(delta2)
			dw1 = (batch_x.T).dot(delta2)
			db1 = np.sum(delta2, axis=0, keepdims=True)

			# add regularization terms
			dw2 += reg_lambda * w2
			dw1 += reg_lambda * w1

			# gradient descent parameter update
			w1 += -epsilon * dw1
			b1 += -epsilon * db1
			w2 += -epsilon * dw2
			b2 += -epsilon * db2

			# update the model
			model = { 'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2 }

			lower += batch_sz
			upper += batch_sz

		if i % 100 == 0:
			print "Training iteration %i" % i

	return model

model = build_model(3, epochs=1000, batch_sz=40)

# plot_decision_boundary(lambda x: predict(model, x))
# plt.show()

results = predict(model, validation_x)

matched = 0
mismatched = 0
for i in range(validation_set_sz):
	if results[i] == validation_y[i]:
		matched += 1
	else:
		mismatched += 1
print "Error: %.4f" % (float(mismatched) / len(validation_y))


















