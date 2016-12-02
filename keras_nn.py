from __future__ import print_function
import numpy as np
import pandas
import sqlite3
import random
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import time
import warnings

#warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#warnings.simplefilter(action = "ignore", category = FutureWarning)

def scale_to_range(new_min, new_max, v):
	new_v = v.copy()
	old_min = min(v)
	old_max = max(v)
	old_range = old_max - old_min
	new_range = new_max - new_min
	for i in range(len(v)):
		new_v[i] = (((v[i] - old_min) * new_range) / old_range) + new_min
	return new_v

def scale_column(scalar, v):
	return v*scalar

def nonnull(a, null_label):
	if(a > null_label):
		return 1
	else:
		return 0

def bin_data(v, col, no_bins=5):
	bin_size = round(100./no_bins)
	percentiles = []
	last_upper_limit = 0
	while(last_upper_limit < 100):
		percentiles.append(last_upper_limit + bin_size)
		last_upper_limit += bin_size
	percentiles = percentiles[:-1]
	for p in range(len(percentiles)):
		percentiles[p]  = round(np.percentile(v[:,col], percentiles[p]))
	percentiles.append(np.max(v[:,col]))
	for i in range(len(v)):
		for j in range(len(percentiles)):
			if(v[i,col] <= percentiles[j]):
				v[i,col] = j+1
				break

def pull_data(skip_ambiguous=False, output_mode=1):
	inputs = []
	outputs = []
	sqlite_file = 'data.db'
	c_type1 = "BYS87"
	c_type2 = "F1S18"
	conn = sqlite3.connect(sqlite_file)
	c = conn.cursor()
	c.execute('SELECT * FROM student WHERE {c1}A>0 AND {c1}C>0 AND {c1}F>0 AND {c2}A>0 AND {c2}B>0 AND {c2}C>0 AND {c2}D>0 AND {c2}E>0 AND {c3}>-1 AND {c4}>-1 AND {c5}>-1 AND {c6}>-1'.\
			format(c1=c_type1, c2=c_type2, c3='F2PS1AID', c4='F3TZSTEM1TOT', c5='F3TZSTEM2GPA', c6='CREDGRAD'))
	all_rows = c.fetchall()
	skip_row = True
	for row in all_rows:
		# if(skip_ambiguous):
		# 	c.execute('SELECT DISTINCT F3TZSTEM2GPA, CREDGRAD FROM student WHERE {c1}A={v1} AND {c1}C={v2} AND {c1}F={v3} AND {c2}A={v4} AND {c2}B={v5} AND {c2}C={v6} AND {c2}D={v7} AND {c2}E={v8} AND {c3}={v9} AND {c4}={v10}'.\
		# 	format(c1=c_type1, c2=c_type2, c3='F2PS1AID', c4='F3TZSTEM1TOT', v1=row[1], v2=row[2], v3=row[3], v4=row[4], v5=row[5], v6=row[6], v7=row[7], v8=row[8], v9=row[9], v10=row[10]))
		# 	outcome_rows = c.fetchall()
		# 	if(len(outcome_rows) > 1):
		# 		continue
		if(skip_ambiguous):
			avg = (row[1]+row[2]+row[3]+row[4]+row[5]+row[6]+row[7]+row[8])/8.
			if(avg == 1 or avg == 4):
				skip_row = False
		if(not skip_ambiguous or (skip_ambiguous and not skip_row)):
			inputs.append([row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10]])
			if(output_mode == 1):
				outputs.append([row[11], row[12]])
			elif(output_mode == 2):
				outputs.append([row[11]])
			elif(output_mode == 3):
				outputs.append([row[12]])
	conn.close()
	inputs = np.asarray(inputs)
	outputs = np.asarray(outputs)
	return (inputs.astype(float), outputs.astype(float))

def define_encoding_classes(column, ordinal=True):
	# define classes and their one-of-K equivalent
	class_list = np.unique(column)
	classes = {}
	for i in range(len(class_list)):
		temp = np.zeros((len(class_list)))
		for j in range(i+1):
			if(ordinal or (not ordinal and j==i)):
				temp[j] = 1.
		classes[class_list[i]] = temp
	inverse_classes = {}
	for k, v in classes.iteritems():
		inverse_classes[''.join([str(int(val)) for val in v])] = k
	return (classes, inverse_classes)

def apply_column_encoding(v, classes):
	# translate column v to binarized form
	binarized = np.empty((len(v),len(classes)))
	for i in range(len(v)):
		binarized[i,:] = classes[v[i]]
	return binarized

def baseline_model(input_dimensions, output_dimensions, hidden_layers, hidden_layer_scale, optimizer_fn, loss_fn, hidden_activation_fn, output_activation_fn):
	model = Sequential()
	hidden_layer_dim = round(input_dimensions * hidden_layer_scale)
	for h in range(hidden_layers):
		input_sz = hidden_layer_dim
		if(h == 0):
			input_sz = input_dimensions
		model.add(Dense(hidden_layer_dim, input_dim=input_sz, init='normal', activation=hidden_activation_fn, name=('hidden'+str(h+1))))
	model.add(Dense(output_dimensions, init='normal', activation=output_activation_fn, name='output'))
	model.compile(loss=loss_fn, optimizer=optimizer_fn, metrics=['accuracy'])
	return model

# returns (avg_accuracy, std_dev, model_obj)
def nn_run(input_train, input_test, output_train, output_test, use_mlpclass=False, no_epochs=1000, no_hidden_layers=1, hidden_layer_size_multiplier=1, optimizer_fn='rmsprop', loss_fn='mse', hidden_activation_fn='sigmoid', output_activation_fn='sigmoid'):

	splits = 5
	batch_sz = 10

	(temp,in_dim) = input_train.shape
	out_dim = 1
	if(output_train.ndim > 1):
		(temp,out_dim) = output_train.shape

	if(use_mlpclass):
		score_kfold = True
		estimator = KerasClassifier(build_fn=baseline_model, input_dimensions=in_dim, output_dimensions=out_dim, hidden_layers=no_hidden_layers, hidden_layer_scale=hidden_layer_size_multiplier, optimizer_fn=optimizer_fn, loss_fn=loss_fn, hidden_activation_fn=hidden_activation_fn, output_activation_fn=output_activation_fn, nb_epoch=no_epochs, batch_size=batch_sz, verbose=0)
		if(score_kfold):
			# use k-fold cross-validation
			kfold = KFold(n_splits=splits, shuffle=True) # removed: random_state=seed
			results = cross_val_score(estimator, input_train, output_train, cv=kfold)
			return (results.mean(), results.std(), estimator)
		else:
			estimator.fit(input_train, output_train)
			return (estimator.score(input_test, output_test), estimator)
	else:
		model = baseline_model(in_dim, out_dim, no_hidden_layers, hidden_layer_size_multiplier, optimizer_fn, loss_fn, hidden_activation_fn, output_activation_fn)
		model.fit(input_train, output_train, nb_epoch=no_epochs, batch_size=batch_sz, verbose=0)
		scores = model.evaluate(input_test, output_test, verbose=0) # scores[0] = loss, scores[1] = accuracy
		return (scores[1], None, model)

def get_weights(model, layer_name):
	wb = model.get_layer(name=layer_name).get_weights()
	return (wb[0], wb[1]) # (weights, biases)

def decode_binarized(v, encoded_class):
	str_v = ''.join([str(int(round(val))) for val in v])
	if(str_v in encoded_class):
		return encoded_class[str_v]
	else:
		return -1

def make_predictions(model, inputs, outputs, test_indices, col1_length, show_output=False, categorical=True, labels=None, output_mode=1):
	matches = 0
	attempts = 0
	sum_delta = 0
	for t_i in test_indices:
		test_input = inputs[t_i,:]
		test_output = None
		if(outputs.ndim > 1):
			test_output = outputs[t_i,:]
		else:
			test_output = outputs[t_i]
		predictions = model.predict(test_input.reshape(1,10))
		if(categorical):
			if(show_output):
				if(output_mode == 1):
					# binary
					# print actual
					print("   actual: ", end="")
					for x in test_output:
						print(" %4d" % int(x), end="")
					if(labels != None):
						print(" [%.2f, %d]" % (decode_binarized(test_output[0:col1_length], labels), test_output[col1_length]))
					else:
						print("\n", end="")
					# print predicted (rounded)
					print("predicted: ", end="")
					for x in predictions[0]:
						print(" %4d" % round(x), end="")
					if(labels != None):
						print(" [%.2f, %d]" % (decode_binarized(predictions[0, 0:col1_length], labels), predictions[0,col1_length]))
					else:
						print("\n", end="")
					# print predicted (float)
					print("predicted: ", end="")
					for x in predictions[0]:
						print(" %.2f" % x, end="")
					print("\n")
				elif(output_mode == 2 or output_mode == 3):
					print("   actual: ", end="")
					for x in test_output:
						print("%4d  " % x, end="")
					print("\npredicted: ", end="")
					for x in predictions[0]:
						print("%4d  " % round(x), end="")
					print("\npredicted: ", end="")
					for x in predictions[0]:
						print("%.2f  " % x, end="")
					print("\n")
			for i in range(len(test_output)):
				if(test_output[i] == round(predictions[0][i])):
					matches += 1
			attempts += len(test_output)
		else:
			if(show_output):
				if(output_mode == 1):
					# continuous
					print("   actual: [", end="")
					spacer = False
					for t_o in test_output:
						if(spacer):
							print(", ", end="")
						print("%.4f" % t_o, end="")
						spacer = True
					print("] [%.4f, %.4f]" % (test_output[0]*4., test_output[1]))
					print("predicted: [", end="")
					spacer = False
					for p in range(len(predictions[0])):
						if(spacer):
							print(", ", end="")
						print("%.4f" % predictions[0][p], end="")
						spacer = True
						sum_delta += abs(predictions[0][p] - test_output[p])
					print("] [%.4f, %.4f]\n" % (predictions[0][0]*4., predictions[0][1]))
				elif(output_mode == 2 or output_mode == 3):
					sum_delta += abs(predictions[0] - test_output)
					print("   actual: %.4f" % test_output)
					print("predicted: %.4f\n" % predictions[0])
			attempts += 1
	acc = 0.0
	if(categorical):
		acc = float(matches)/attempts
	else:
		acc = sum_delta/attempts
	return acc

# ==================================================================================
# ==================================================================================
# ==================================================================================

# loops:
# 	categorical, regression
# 	output_mode 1, 2, 3
# 	epochs 10 50 200 500
# ...or just capture samples

# seed = 7
# np.random.seed(seed)

validation_proportion = 0.25
skip_ambiguous = True
epochs = 10
categorical = True
no_tests = 20
output_mode = 2
# 1 = GPA, POSTGRAD (2 targets)
# 2 = GPA (1 target)
# 3 = POSTGRAD (1 target)

test_data = False

inputs = None
outputs = None
input_train = None
input_test = None
output_train = None
output_test = None
encoded_input_train = None
encoded_output_train = None
encoded_input_test = None
encoded_output_test = None
if(test_data):
	dataframe = pandas.read_csv("data/mushroom.csv", header=None)
	dataset = dataframe.values
	np.random.shuffle(dataset)
	inputs = dataset[:,1:]
	outputs = dataset[:,0]
	input_classes = []
	inverse_input_classes = []
	for i in range(len(inputs[0])):
		(temp_classes, temp_inverse_classes) = define_encoding_classes(inputs[:,i], ordinal=False)
		input_classes.append(temp_classes)
		inverse_input_classes.append(temp_inverse_classes)
	output_classes, inverse_output_classes = define_encoding_classes(outputs, ordinal=False)
	input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size=validation_proportion)
	# TRAIN INPUT
	for i in range(len(input_train[0])):
		encoded_col = apply_column_encoding(input_train[:,i], input_classes[i])
		if(encoded_input_train == None):
			encoded_input_train = encoded_col
		else:
			encoded_input_train = np.column_stack((encoded_input_train, encoded_col))
	# TRAIN OUTPUT
	encoded_col = apply_column_encoding(output_train, output_classes)
	if(encoded_output_train == None):
		encoded_output_train = encoded_col
	else:
		encoded_output_train = np.column_stack((encoded_output_train, encoded_col))
	# TEST INPUT
	for i in range(len(input_test[0])):
		encoded_col = apply_column_encoding(input_test[:,i], input_classes[i])
		if(encoded_input_test == None):
			encoded_input_test = encoded_col
		else:
			encoded_input_test = np.column_stack((encoded_input_test, encoded_col))
	# TEST OUTPUT
	encoded_col = apply_column_encoding(output_test, output_classes)
	if(encoded_output_test == None):
		encoded_output_test = encoded_col
	else:
		encoded_output_test = np.column_stack((encoded_output_test, encoded_col))
else:
	(inputs, outputs) = pull_data(skip_ambiguous, output_mode)
	print(inputs.shape)
	print(outputs.shape)
	bin_data(inputs, 9) # bin no. STEM courses
	ordinal = True
	if(output_mode == 3):
		ordinal = False
	(classes, inverse_classes) = define_encoding_classes(outputs[:,0], ordinal)
	input_train, input_test, output_train, output_test = train_test_split(inputs, outputs, test_size=validation_proportion) # removed: random_state=seed
	if(categorical):
		# TRAIN - one-hot encode output
		encoded_output_train_col1 = apply_column_encoding(output_train[:,0], classes)
		encoded_output_train = None
		if(output_mode == 1):
			encoded_output_train = np.column_stack((encoded_output_train_col1, output_train[:,1]))
		elif(output_mode == 2 or output_mode == 3):
			encoded_output_train = encoded_output_train_col1
		# TEST - one-hot encode output
		encoded_output_test_col1 = apply_column_encoding(output_test[:,0], classes)
		encoded_output_test = None
		if(output_mode == 1):
			encoded_output_test = np.column_stack((encoded_output_test_col1, output_test[:,1]))
		elif(output_mode == 2 or output_mode == 3):
			encoded_output_test = encoded_output_test_col1

if(categorical):
	if(test_data):
		test_indices = np.random.randint(len(encoded_input_test), size=no_tests)
		# build NN
		loss_fn = 'categorical_crossentropy'
		(mu, sigma, model) = nn_run(encoded_input_train, encoded_input_test, encoded_output_train, encoded_output_test, hidden_activation_fn='sigmoid', output_activation_fn='sigmoid', no_epochs=epochs, loss_fn=loss_fn)
		print("Acc (reported by binary cross entropy): %.4f\n" % mu)
		# predict
		matched = 0
		attempts = 0
		for i in test_indices:
			predictions = model.predict(encoded_input_test[i,:].reshape(1,encoded_input_test.shape[1]))
			print("   actual: ", end="")
			print([int(round(j)) for j in encoded_output_test[i,:]])
			print("predicted: ", end="")
			print([int(round(j)) for j in predictions[0]])
			print("\n", end="")
			for j in range(len(encoded_output_test[i,:])):
				if(encoded_output_test[i,j] == round(predictions[0][j])):
					matched += 1
				attempts += 1
		print("Acc: %.4f" % (float(matched)/attempts))
	else:
		test_indices = np.random.randint(len(input_test), size=no_tests)
		# build NN
		loss_fn = 'categorical_crossentropy'
		if(output_mode == 3):
			loss_fn = 'binary_crossentropy'
		(mu, sigma, model) = nn_run(input_train, input_test, encoded_output_train, encoded_output_test, hidden_activation_fn='sigmoid', output_activation_fn='sigmoid', no_epochs=epochs, loss_fn=loss_fn)
		print("Acc (reported by cat. cross entropy): %.4f\n" % mu)
		# predict
		acc = make_predictions(model, input_test, encoded_output_test, test_indices, len(classes), True, True, inverse_classes, output_mode)
		print("Acc (%% indices matched): %.4f" % acc)
else:
	test_indices = np.random.randint(len(input_test), size=no_tests)
	# regression/continuous
	# TRAIN - scale output to 0..1
	scaled_output_train_col1 = scale_to_range(0., 1., output_train[:,0])
	scaled_output_train_col1 = scale_column(0.25, output_train[:,0])
	scaled_output_train = None
	if(output_mode == 1):
		scaled_output_train = np.column_stack((scaled_output_train_col1, output_train[:,1]))
	elif(output_mode == 2 or output_mode == 3):
		scaled_output_train = scaled_output_train_col1
	# TEST - scale output to 0..1
	scaled_output_test_col1 = scale_to_range(0., 1., output_test[:,0])
	scaled_output_test_col1 = scale_column(0.25, output_test[:,0])
	scaled_output_test = None
	if(output_mode == 1):
		scaled_output_test = np.column_stack((scaled_output_test_col1, output_test[:,1]))
	else:
		scaled_output_test = scaled_output_test_col1

	# build NN
	(mu, sigma, model) = nn_run(input_train, input_test, scaled_output_train, scaled_output_test, no_epochs=epochs, loss_fn='mse')
	print("Acc (MSE): %.4f\n" % mu)
	# predict
	acc = make_predictions(model, input_test, scaled_output_test, test_indices, 1, True, False, None, output_mode)
	print("Avg. error: %.4f" % acc)


# why identical/frozen predictions on categorical?

# * scale all targets to 0..1
# * accuracy all - average distance actual, predicted
# regression (sigmoid / sigmoid + scale / MSE)
# 	[X] GPA, POSTGRAD
# 	[X] GPA
# 	[X] POSTGRAD
# 	[-] GPA + POSTGRAD

# * accuracy - binary matching
# multi class (sigmoid / sigmoid or softmax / categorical_CE)
# 	[X] GPA, POSTGRAD
# 	[X] GPA
# 	[-] GPA + POSTGRAD

# * accuracy - binary matching
# binary class (sigmoid / sigmoid / binary_CE)
# 	[X] POSTGRAD
# 