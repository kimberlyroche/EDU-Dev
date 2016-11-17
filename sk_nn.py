import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm

def scale_to_range(min_old, max_old, min_new, max_new, value):
	old_range = (max_old - min_old)
	new_range = (max_new - min_new)
	return (((value - min_old) * new_range) / float(old_range)) + min_new

filename = '/Users/kimroche/Desktop/STEM_only_clean_20160902_LOCAL.csv'
color_scale = 3

def read_nn_input(filename):
	input = []
	output = []

	f = open(filename, 'rb')
	discarded = 0
	retained = 0
	skip = True
	for line in f:
		if skip:
			skip = False
			continue;
		cleaned = line.strip().split(',')
		if "999" not in cleaned:
			input.append([ \
				float(cleaned[1]), float(cleaned[2]), float(cleaned[3]), float(cleaned[4]), \
				float(cleaned[5]), float(cleaned[6]), float(cleaned[7]), float(cleaned[8]), \
				float(cleaned[9]), float(cleaned[10]), float(cleaned[11]), float(cleaned[12]), \
				float(cleaned[13]), float(cleaned[14]), float(cleaned[15]), float(cleaned[16]), \
			])
			output.append([ \
				float(cleaned[17]), float(cleaned[18]), float(cleaned[19]), float(cleaned[20]) \
			])
			retained += 1
		else:
			discarded += 1

	print "RETAINED: " + str(retained)
	print "DISCARDED: " + str(discarded)

	return (input, output)

def bound_data(old_min, old_max, vector):
	new_min = -3
	new_max = 3
	for i in range(len(vector)):
		vector[i] = scale_to_range(old_min, old_max, new_min, new_max, vector[i])
	return vector

def fit_nn(input, output):
	clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
	# clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
	clf.fit(input, output)
	# print clf
	predicted = []
	pred_min = 0.
	pred_max = 0.
	for i in range(len(input)):
		predicted_row = clf.predict([input[i]])[0]
		pred_min_row = min(predicted_row)
		if pred_min_row < pred_min:
			pred_min = pred_min_row
		pred_max_row = max(predicted_row)
		if pred_max_row > pred_max:
			pred_max = pred_max_row
		predicted.append(bound_data(pred_min, pred_max, predicted_row))
	return pred_min, pred_max, predicted

input, output = read_nn_input(filename)

input = np.asarray(input)
output = np.asarray(output)

input_mins = input.min(axis=0)
input_maxes = input.max(axis=0)
output_mins = output.min(axis=0)
output_maxes = output.max(axis=0)

input = input.T
rows, cols = input.shape
for i in range(rows):
	input[i] = bound_data(input_mins[i], input_maxes[i], input[i,:])
input = input.T

output = output.T
rows, cols = output.shape
for i in range(rows):
	output[i] = bound_data(output_mins[i], output_maxes[i], output[i,:])
output = output.T

total_map = input.T
pred_min, pred_max, predicted = fit_nn(input, output)
predicted = np.asarray(predicted)
predicted = predicted.T
rows, cols = total_map.shape
total_map = np.append(total_map, np.zeros([1,cols]), axis=0)
total_map = np.append(total_map, predicted, axis=0)
total_map = np.append(total_map, np.zeros([1,cols]), axis=0)
total_map = np.append(total_map, output.T, axis=0)

total_map = total_map.T

if abs(pred_min) > abs(pred_max) and abs(pred_min) > color_scale:
	color_scale = abs(pred_min)
else:
	if abs(pred_max) > abs(pred_min) and abs(pred_max) > color_scale:
		color_scale = abs(pred_max)

fig = plt.imshow(total_map[0:30,:], cmap=cm.seismic, interpolation='nearest', vmin=-color_scale, vmax=color_scale)
plt.colorbar()
plt.show()



