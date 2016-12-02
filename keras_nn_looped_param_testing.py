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

# =========================================================================
# =========================================================================
#                            DATA MANAGEMENT STUFF
# =========================================================================
# =========================================================================

def scale_to_range(OldMin, OldMax, NewMin, NewMax, OldValue):
	OldRange = (OldMax - OldMin)
	NewRange = (NewMax - NewMin)
	NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
	return NewValue

def nonnull(a, null_label):
	if(a > null_label):
		return 1
	else:
		return 0

def get_mean_bycols(v, null_label):
	mean_bycol = []
	sum_bycol = v.sum(axis=0)
	vecfunc = np.vectorize(nonnull)
	membership_bycol = vecfunc(v, null_label)
	for i in range(len(sum_bycol)):
		col_content = membership_bycol[:,i]
		mean_bycol.append(np.round(float(sum_bycol[i])/col_content.sum()))
	return mean_bycol

# impute from column mean
def impute1(v, null_label):
	mean_bycol = get_mean_bycols(v, null_label)
	for i in range(len(v)):
		for j in range(len(v[i])):
			if v[i][j] == null_label:
				v[i][j] = int(mean_bycol[j])

# find rows matching all columns in the target row but with extant data
# in the columns (ignore_cols) where the target has nulls
def find_matches(v, target, ignore_cols, null_label):
	match_idx_list = []
	for i in range(len(v)):
		found = True
		for j in range(len(v[i])):
			if(j in ignore_cols and v[i][j] == null_label):
				found = False
				break
			elif(j not in ignore_cols and v[i][j] != target[j]):
				found = False
				break
		if(found):
			match_idx_list.append(i)
	return match_idx_list

# impute more elaborately, using "state-aware" column mean
# m = 1
# for rows missing m values
# 	randomly choose a column to search from the missing ones
#   find identical columns masking on missing rows
#   randomly choose from among these a row whose column to steal
# m++
def impute2(v, null_label):
	verbose = False
	m_of_interest = 3
	mean_bycols = get_mean_bycols(v, null_label)
	row_list = []
	for m in range(8):
		# print("Imputing data for rows missing " + str(m) + " elements...")
		if(m == 0):
			continue
		count = 0
		for i in range(len(v)):
			missing = []
			for j in range(len(v[i])):
				if(v[i][j] == null_label):
					missing.append(j)
			if(len(missing) == m):
				match_idx_list = find_matches(v, v[i], missing, null_label)
				if(len(match_idx_list) == 0):
					for k in missing:
						v[i][k] = mean_bycols[k]
				else:
					for k in missing:
						rand_idx = random.randint(0, len(match_idx_list)-1)
						v[i][k] = v[match_idx_list[rand_idx]][k]

def pull_data(impute_inputs=False, impute_outputs=False, impute_method=1):
	inputs = []
	outputs = []
	sqlite_file = 'data.db'
	c_type1 = "BYS87"
	c_type2 = "F1S18"
	conn = sqlite3.connect(sqlite_file)
	c = conn.cursor()
	if(impute_inputs and not impute_outputs):
		# require some non-null input and all non-null output
		c.execute('SELECT * FROM student WHERE ({c1}A>0 OR {c1}C>0 OR {c1}F>0 OR {c2}A>0 OR {c2}B>0 OR {c2}C>0 OR {c2}D>0 OR {c2}E>0) AND {c3}>-1 AND {c4}>-1 AND {c5}>-1 AND {c6}>-1'.\
			format(c1=c_type1, c2=c_type2, c3='F2PS1AID', c4='F3TZSTEM1TOT', c5='F3TZSTEM2GPA', c6='CREDGRAD'))
	elif(impute_inputs and impute_outputs):
		# require some non-null input and some non-null output
		c.execute('SELECT * FROM student WHERE ({c1}A>0 OR {c1}C>0 OR {c1}F>0 OR {c2}A>0 OR {c2}B>0 OR {c2}C>0 OR {c2}D>0 OR {c2}E>0) AND ({c3}>-1 OR {c4}>-1 OR {c5}>-1 OR {c6}>-1)'.\
			format(c1=c_type1, c2=c_type2, c3='F2PS1AID', c4='F3TZSTEM1TOT', c5='F3TZSTEM2GPA', c6='CREDGRAD'))
	elif(not impute_inputs and not impute_outputs):
		# require all non-null input and all non-null output
		c.execute('SELECT * FROM student WHERE ({c1}A>0 AND {c1}C>0 AND {c1}F>0 AND {c2}A>0 AND {c2}B>0 AND {c2}C>0 AND {c2}D>0 AND {c2}E>0) AND ({c3}>-1 AND {c4}>-1 AND {c5}>-1 AND {c6}>-1)'.\
			format(c1=c_type1, c2=c_type2, c3='F2PS1AID', c4='F3TZSTEM1TOT', c5='F3TZSTEM2GPA', c6='CREDGRAD'))
	elif(not impute_inputs and impute_outputs):
		# require all non-null input and some non-null output
		c.execute('SELECT * FROM student WHERE ({c1}A>0 AND {c1}C>0 AND {c1}F>0 AND {c2}A>0 AND {c2}B>0 AND {c2}C>0 AND {c2}D>0 AND {c2}E>0) AND ({c3}>-1 OR {c4}>-1 OR {c5}>-1 OR {c6}>-1)'.\
			format(c1=c_type1, c2=c_type2, c3='F2PS1AID', c4='F3TZSTEM1TOT', c5='F3TZSTEM2GPA', c6='CREDGRAD'))
	all_rows = c.fetchall()
	for row in all_rows:
		inputs.append([row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]])
		outputs.append([row[9], row[10], row[11], row[12]])
	conn.close()
	inputs = np.asarray(inputs)
	outputs = np.asarray(outputs)
	if(impute_inputs):
		if(impute_method == 1):
			impute1(inputs, 0)
		else:
			impute2(inputs, 0)

	if(impute_outputs):
		if(impute_method == 1):
			impute1(outputs, -1)
		else:
			impute2(outputs, -1)
	return (inputs, outputs)

def bin_data(v, no_bins=5):
	bin_size = round(100./no_bins)
	percentiles = []
	last_upper_limit = 0
	while(last_upper_limit < 100):
		percentiles.append(last_upper_limit + bin_size)
		last_upper_limit += bin_size
	percentiles = percentiles[:-1]
	for p in range(len(percentiles)):
		percentiles[p]  = round(np.percentile(v, percentiles[p]))
	percentiles.append(np.max(v))
	for i in range(len(v)):
		for j in range(len(percentiles)):
			if(outputs[i,1] <= percentiles[j]):
				outputs[i,1] = j+1
				break

def binarize_variable(v, ordinal=True):
	# define classes and their one-hot equivalent
	class_list = np.unique(v)
	classes = {}
	for i in range(len(class_list)):
		temp = np.zeros((len(class_list)))
		for j in range(i+1):
			if(ordinal or (not ordinal and j==i)):
				temp[j] = 1.
		classes[class_list[i]] = temp
	# translate column v to binarized form
	binarized = np.empty((len(v),len(classes)))
	for i in range(len(v)):
		binarized[i,:] = classes[v[i]]
	inverse_classes = {}
	for k, v in classes.iteritems():
		inverse_classes[''.join([str(int(val)) for val in v])] = k
	return (binarized, inverse_classes)


# =========================================================================
# =========================================================================
#                           KERAS NN STUFF
# =========================================================================
# =========================================================================

def baseline_model(input_dimensions, output_dimensions, hidden_layers=1, hidden_layer_scale=1.5, optimizer_fn='sgd', loss_fn='mse', hidden_activation_fn='sigmoid', output_activation_fn='sigmoid'):
	model = Sequential()
	hidden_layer_dim = round(input_dimensions * hidden_layer_scale)
	for h in range(hidden_layers):
		input_sz = hidden_layer_dim
		if(h == 0):
			input_sz = input_dimensions
		model.add(Dense(hidden_layer_dim, input_dim=input_sz, init='normal', activation=hidden_activation_fn, name=('hidden'+str(h+1))))
	# FIRST TESTING RUN -- model.add(Dense(output_dimensions, init='normal', activation='sigmoid', name='output'))
	model.add(Dense(output_dimensions, init='normal', activation=output_activation_fn, name='output'))
	model.compile(loss=loss_fn, optimizer=optimizer_fn, metrics=['accuracy'])
	return model

def one_hot_encode(v):
	encoded = None
	if(v.ndim > 1):
		encoded_columns = 0
		for column in v.T:
			encoded_columns += len(set(column))
		encoded = np.empty((len(v), encoded_columns))
		fits = []
		last_column_populated = -1
		for row in v.T:
			encoder = LabelEncoder()
			encoder.fit(row)
			encoded_col = encoder.transform(row)
			dummy_col = np_utils.to_categorical(encoded_col)
			(encoded_row_no, encoded_col_no) = dummy_col.shape
			if(last_column_populated < 0):
				encoded[:,:encoded_col_no] = dummy_col[:,:]
				last_column_populated = encoded_col_no-1
			else:
				encoded[:,(last_column_populated+1):(last_column_populated+encoded_col_no+1)] = dummy_col
				last_column_populated += encoded_col_no
	else:
		encoder = LabelEncoder()
		encoder.fit(v)
		encoded = encoder.transform(v)
		encoded = np_utils.to_categorical(encoded)
	return encoded

def nn_run(inputs, outputs, use_mlpclass=False, no_hidden_layers=1, hidden_layer_size_multiplier=1.5, optimizer_fn='sgd', loss_fn='mse', hidden_activation_fn='sigmoid', output_activation_fn='sigmoid'):
	# fixed parameters
	# 10 epochs ~= 1 second
	no_epoch = 500
	# no_epoch = 20
	splits = 5
	batch_sz = 10
	validation_proportion = 0.2
	# dataset_upper_limit = 500 # REMOVE
	# X = inputs[:dataset_upper_limit,:] # REMOVE
	# Y = outputs[:dataset_upper_limit,:] # REMOVE

	X = inputs
	Y = outputs

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_proportion, random_state=seed)
	(temp,in_dim) = X_train.shape
	out_dim = 1
	if(Y_train.ndim > 1):
		(temp,out_dim) = Y_train.shape

	if(use_mlpclass):
		score_kfold = True
		estimator = KerasClassifier(build_fn=baseline_model, input_dimensions=in_dim, output_dimensions=out_dim, hidden_layers=no_hidden_layers, hidden_layer_scale=hidden_layer_size_multiplier, optimizer_fn=optimizer_fn, loss_fn=loss_fn, hidden_activation_fn=hidden_activation_fn, output_activation_fn=output_activation_fn, nb_epoch=no_epoch, batch_size=batch_sz, verbose=0)
		if(score_kfold):
			# use k-fold cross-validation to "describes the evaluation of the 10 
			# 		constructed models for each of the splits of the dataset"
			kfold = KFold(n_splits=splits, shuffle=True, random_state=seed) # 10-fold cross-validation
			results = cross_val_score(estimator, X, Y, cv=kfold)
			return (results.mean(), results.std(), estimator)
			# print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))
		else:
			estimator.fit(X_train, Y_train)
			return (estimator.score(X_test, Y_test), None, estimator)
	else:
		model = baseline_model(in_dim, out_dim, hidden_layers=no_hidden_layers, hidden_layer_scale=hidden_layer_size_multiplier, optimizer_fn=optimizer_fn, loss_fn=loss_fn, hidden_activation_fn=hidden_activation_fn, output_activation_fn=output_activation_fn)
		model.fit(X_train, Y_train, nb_epoch=no_epoch, batch_size=batch_sz, verbose=0)
		scores = model.evaluate(X_test, Y_test, verbose=0)
		# loss is scores[0], accuracy is scores[1]
		return (scores[1], None, model, X_test, Y_test)

		# wb = model.get_layer(name='hidden').get_weights()
		# wb[0] = weights, wb[1] = biases

def get_encoded_inputs(inputs):
	encoded_inputs = None
	input_shape = inputs.shape
	for i in range(input_shape[1]):
		(encoded_col, col_classes) = binarize_variable(inputs[:,i])
		if(encoded_inputs == None):
			encoded_inputs = encoded_col
		else:
			encoded_inputs = np.append(encoded_inputs, encoded_col, axis=1)
	return encoded_inputs

def get_encoded_outputs(outputs, encode_binary=True):
	encoded_outputs = None
	output_classes = []
	output_shape = outputs.shape
	for i in range(output_shape[1]):
		encoded_col = None
		col_classes = None
		if((i > 0 and i < 3) or encode_binary):
			(encoded_col, col_classes) = binarize_variable(outputs[:,i], ordinal=True)
			output_classes.append(col_classes)
		else:
			encoded_col = outputs[:,i]
			output_classes.append({})
		if(encoded_outputs == None):
			encoded_outputs = encoded_col
		else:
			encoded_outputs = np.column_stack((encoded_outputs, encoded_col))
			# encoded_outputs = np.append(encoded_outputs, encoded_col, axis=1)
	return (encoded_outputs,output_classes)

def decode_binarized(v, encode_class):
	str_v = ''.join([str(int(round(val))) for val in v])
	return encode_class[str_v]


# =========================================================================
# =========================================================================
#                                 MAIN
# =========================================================================
# =========================================================================

seed = 7
np.random.seed(seed)
start_time = time.time()

# test_type:
#	1 = test params (activation/loss/optimization functions + binarization of input/output)
#	2 = test structure (number and size of layers, imputed vs. intact input/output)
#	3 = run specific test
test_type = 3
if(test_type == 1):
	f_out = open('keras_out_params.txt', 'w')

	inputs, outputs = pull_data(impute_inputs=False, impute_outputs=False, impute_method=1)
	bin_data(outputs[:,1]) # bin the continuous # STEM courses 

	encoded_inputs = get_encoded_inputs(inputs)
	encoded_outputs = get_encoded_outputs(outputs)

	f_out.write("encoded_in\tencoded_out\tloss_fn\toptimizer\tactivation_hidden\tactivation_output\taccuracy\tstd_dev\n")
	for a in[False, True]:
		for b in[False, True]:
			for c in ['categorical_crossentropy', 'mean_squared_error']: # categorical CE should be appropriate for classification, MSE for regression
				for d in ['rmsprop']: # not evaluating sgd or other adaptive learning algorithms adam, adadelta, nadam, etc.
					for e in ['sigmoid','relu','hard_sigmoid']: # not evaluating tanh (-1..1) or softmax
						for f in ['sigmoid','relu','hard_sigmoid']:
							(acc, sd, model) = (-1, -1, None)
							copied_inputs = inputs
							copied_outputs = outputs
							if(a):
								copied_inputs = encoded_inputs
							if(b):
								copied_outputs = encoded_outputs
							(acc, sd, model) = nn_run(copied_inputs, copied_outputs, use_mlpclass=True, no_hidden_layers=1, hidden_layer_size_multiplier=1, optimizer_fn=d, loss_fn=c, hidden_activation_fn=e, output_activation_fn=f)
							f_out.write(str(a) + "\t")
							f_out.write(str(b) + "\t")
							f_out.write(str(c) + "\t")
							f_out.write(str(d) + "\t")
							f_out.write(str(e) + "\t")
							f_out.write(str(f) + "\t")
							f_out.write("%.4f\n" % acc)
							print(str(a) + "\t", end="")
							print(str(b) + "\t", end="")
							print(str(c) + "\t", end="")
							print(str(d) + "\t", end="")
							print(str(e) + "\t", end="")
							print(str(f) + "\t", end="")
							print("%.4f\t%.4f\n" % (acc,sd), end="")
	f_out.close()
elif(test_type == 2):
	f_out = open('keras_out_structure.txt', 'w')
	f_out.write("input_inputs\timpute_outputs\timpute_method\tcross_validate\thidden_layers\thidden_layer_size\twhich_data\taccuracy\truns\tstd_dev\n")
	for a in [False, True]:
		for b in [False, True]:
			for c in [1, 2]:
				for d in [True]:
					for e in [1, 2]:
						for f in [1, 1.5, 2]:
							for g in [3]:
								# 0 = test input, 4 x 1 (super easy)
								# 1 = test input, 22 x 1
								# 2 = test input, 8 x 4
								# 3 = real input, 8 x 4
								inputs = None
								outputs = None
								encoded_inputs = None
								encoded_outputs = None
								if(g == 0):
									# test dataset, output must be binary encoded
									dataframe = pandas.read_csv("iris.csv", header=None)
									dataset = dataframe.values
									np.random.shuffle(dataset)
									inputs = dataset[:,0:4].astype(float)
									outputs = dataset[:,4]
									encoded_inputs = inputs
									encoded_outputs = one_hot_encode(outputs)
								elif(g == 1):
									# test dataset, labels must be binary encoded
									dataframe = pandas.read_csv("mushroom.csv", header=None)
									dataset = dataframe.values
									np.random.shuffle(dataset)
									inputs = dataset[:,0:22]
									outputs = dataset[:,22:]
									encoded_inputs = one_hot_encode(inputs)
									encoded_outputs = one_hot_encode(outputs)
								elif(g == 2):
									# test dataset, labels must be binary encoded
									dataframe = pandas.read_csv("mushroom.csv", header=None)
									dataset = dataframe.values
									np.random.shuffle(dataset)
									inputs = dataset[:,0:8]
									outputs = dataset[:,-4:]
									encoded_inputs = one_hot_encode(inputs)
									encoded_outputs = one_hot_encode(outputs)
								else:
									inputs, outputs = pull_data(impute_inputs=a, impute_outputs=b, impute_method=c)
									bin_data(outputs[:,1]) # bin the continuous # STEM courses variable
									encoded_inputs = get_encoded_inputs(inputs)
									encoded_outputs = get_encoded_outputs(outputs)
								# skippable cases: imputing input or output while using test data
								if((a or b) and (g < 3)):
									(acc, sd, model) = (-1, -1, None)
								else:
									(acc, sd, model) = nn_run(encoded_inputs, encoded_outputs, use_mlpclass=d, no_hidden_layers=e, hidden_layer_size_multiplier=f, optimizer_fn='adam', loss_fn='categorical_crossentropy', hidden_activation_fn='relu', output_activation_fn='relu')
								f_out.write(str(a) + "\t")
								f_out.write(str(b) + "\t")
								f_out.write(str(c) + "\t")
								f_out.write(str(d) + "\t")
								f_out.write(str(e) + "\t")
								f_out.write(str(f) + "\t")
								f_out.write(str(g) + "\t")
								print(str(a) + "\t", end="")
								print(str(b) + "\t", end="")
								print(str(c) + "\t", end="")
								print(str(d) + "\t", end="")
								print(str(e) + "\t", end="")
								print(str(f) + "\t", end="")
								print(str(g) + "\t", end="")
								if(d == False):
									# only one run, no std dev recorded
									f_out.write("%0.3f\t1\tNA\n" % acc)
									print("%0.3f\t1\tNA\n" % acc, end="")
								else:
									f_out.write("%0.3f\t5\t%.3f\n" % (acc, sd))
									print("%0.3f\t5\t%.3f\n" % (acc, sd), end="")
else:
	inputs, outputs = pull_data(impute_inputs=False, impute_outputs=False, impute_method=1)
	bin_data(outputs[:,1]) # bin the continuous # STEM courses 
	# outputs = np.column_stack((outputs[:,0], outputs[:,3]))

	encode_binary = True
	encoded_inputs = get_encoded_inputs(inputs)
	(encoded_outputs, output_classes) = get_encoded_outputs(outputs, encode_binary)

	(acc, sd, model, encoded_inputs, encoded_outputs) = nn_run(encoded_inputs, encoded_outputs, use_mlpclass=False, no_hidden_layers=1, hidden_layer_size_multiplier=1, optimizer_fn='rmsprop', loss_fn='mean_squared_error', hidden_activation_fn='sigmoid', output_activation_fn='hard_sigmoid')
	print("Acc: %f\n" % acc)

	wb = model.get_layer(name='hidden1').get_weights()
	print("WEIGHTS:")
	for i in range(len(wb[0][0])):
		print("%.4f  " % wb[0][0][i], end="")
	print("\nBIASES:\n", end="")
	for i in range(len(wb[1])):
		print("%.4f  " % wb[1][i], end="")
	print("\n", end="")

	no_tests = 20
	(encoded_input_rows, encoded_input_cols) = encoded_inputs.shape
	(encoded_output_rows, encoded_output_cols) = encoded_outputs.shape
	test_indices = np.random.randint(encoded_input_rows, size=no_tests)
	acc_binary = [0., 0.]
	acc_cat = [0., 0.]
	for i in range(no_tests):
		test_idx = test_indices[i]
		test_input = encoded_inputs[test_idx,:]
		reshaped_input = test_input.reshape((1, encoded_input_cols))
		predictions = model.predict(reshaped_input)
		print("%4i:   " % test_idx, end="")
		for p in range(encoded_output_cols):
			print("%.0f " % encoded_outputs[test_idx,p], end="")
		print("\n%4i:   " % test_idx, end="")
		for p in range(encoded_output_cols):
			print("%.0f " % round(predictions[0,p]), end="")
			if(encoded_outputs[test_idx,p] == round(predictions[0,p])):
				acc_binary[0] += 1
			acc_binary[1] += 1
		print("\n")
		if(encode_binary):
			# columns (old -> new) are:
			#	0 -> 0,1
			#	1 -> 2,3,4,5,6
			#	2 -> 7,8,9,10,11,12,13,14,15,16,17,18,19
			#	3 -> 20,21
			print("%.1f %.1f %.2f %.1f\n" % (outputs[test_idx, 0], outputs[test_idx, 1], outputs[test_idx, 2], outputs[test_idx, 3]), end="")
			print("%.1f %.1f %.2f %.1f\n" % (decode_binarized(predictions[0, 0:2], output_classes[0]), decode_binarized(predictions[0, 2:7], output_classes[1]), decode_binarized(predictions[0, 7:20], output_classes[2]), decode_binarized(predictions[0, 20:22], output_classes[3])))
			if(outputs[test_idx, 0] == decode_binarized(predictions[0, 0:2], output_classes[0])):
				acc_cat[0] += 1
			acc_cat[1] += 1
			if(outputs[test_idx, 1] == decode_binarized(predictions[0, 2:7], output_classes[1])):
				acc_cat[0] += 1
			acc_cat[1] += 1
			if(outputs[test_idx, 2] == decode_binarized(predictions[0, 7:20], output_classes[2])):
				acc_cat[0] += 1
			acc_cat[1] += 1
			if(outputs[test_idx, 3] == decode_binarized(predictions[0, 20:22], output_classes[3])):
				acc_cat[0] += 1
			acc_cat[1] += 1
		else:
			# columns (old -> new) are:
			#	0 -> 0
			#	1 -> 1,2,3,4,5
			#	2 -> 6,7,8,9,10,11,12,13,14,15,16,17,18
			#	3 -> 19
			print("%.1f %.1f %.2f %.1f\n" % (outputs[test_idx, 0], outputs[test_idx, 1], outputs[test_idx, 2], outputs[test_idx, 3]), end="")
			print("%.1f %.1f %.2f %.1f\n" % (round(predictions[0, 0]), decode_binarized(predictions[0, 1:6], output_classes[1]), decode_binarized(predictions[0, 6:19], output_classes[2]), round(predictions[0, 19])))
			if(outputs[test_idx, 0] == round(predictions[0, 0])):
				acc_cat[0] += 1
			acc_cat[1] += 1
			if(outputs[test_idx, 1] == decode_binarized(predictions[0, 1:6], output_classes[1])):
				acc_cat[0] += 1
			acc_cat[1] += 1
			if(outputs[test_idx, 2] == decode_binarized(predictions[0, 6:19], output_classes[2])):
				acc_cat[0] += 1
			acc_cat[1] += 1
			if(outputs[test_idx, 3] == round(predictions[0, 19])):
				acc_cat[0] += 1
			acc_cat[1] += 1
	print('"Binary" accuracy: %.3f' % (acc_binary[0]/acc_binary[1]))
	print('Categorical accuracy: %.3f' % (acc_cat[0]/acc_cat[1]))

print("time: %s" % (time.time() - start_time))










