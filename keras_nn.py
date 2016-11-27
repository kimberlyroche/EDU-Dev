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

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 
warnings.filterwarnings("ignore", category=np.FutureWarning) 

# =========================================================================
# =========================================================================
#                            DATA MANAGEMENT STUFF
# =========================================================================
# =========================================================================

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
	return binarized


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
	no_epoch = 200
	batch_sz = 10
	validation_proportion = 0.2
	# dataset_upper_limit = 500 # REMOVE
	# X = inputs[:dataset_upper_limit,:] # REMOVE
	# Y = outputs[:dataset_upper_limit,:] # REMOVE

	X = inputs
	Y = outputs

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_proportion, random_state=seed)
	(temp,in_dim) = X_train.shape
	(temp,out_dim) = Y_train.shape

	if(use_mlpclass):
		score_kfold = True
		estimator = KerasClassifier(build_fn=baseline_model, input_dimensions=in_dim, output_dimensions=out_dim, hidden_layers=no_hidden_layers, hidden_layer_scale=hidden_layer_size_multiplier, optimizer_fn='sgd', loss_fn='mse', hidden_activation_fn='sigmoid', output_activation_fn='sigmoid', nb_epoch=no_epoch, batch_size=batch_sz, verbose=0)
		if(score_kfold):
			# use k-fold cross-validation to "describes the evaluation of the 10 
			# 		constructed models for each of the splits of the dataset"
			kfold = KFold(n_splits=5, shuffle=True, random_state=seed) # 10-fold cross-validation
			results = cross_val_score(estimator, X, Y, cv=kfold)
			return (results.mean(), results.std())
			# print("Accuracy: %0.2f (+/- %0.2f)" % (results.mean(), results.std() * 2))
		else:
			estimator.fit(X_train, Y_train)
			# print(estimator.score(X_test, Y_test))
	else:
		model = baseline_model(in_dim, out_dim, hidden_layers=no_hidden_layers, hidden_layer_scale=hidden_layer_size_multiplier, optimizer_fn='sgd', loss_fn='mse', hidden_activation_fn='sigmoid', output_activation_fn='sigmoid')
		model.fit(X_train, Y_train, nb_epoch=no_epoch, batch_size=batch_sz, verbose=0)
		scores = model.evaluate(X_test, Y_test, verbose=0)
		# loss is scores[0], accuracy is scores[1]
		return (scores[1], None)

		# wb = model.get_layer(name='hidden').get_weights()
		# wb[0] = weights, wb[1] = biases

def make_prediction(model, test_input, encoder=None):
	predictions = model.predict(test_input)
	# print(predictions)
	# print([round(x) for x in predictions])
	# if(encoder != None):
	# 	print(encoder.inverse_transform(predictions))


# =========================================================================
# =========================================================================
#                                 MAIN
# =========================================================================
# =========================================================================

test_params = True
if(test_params):
	f_out = open('keras_out_params.txt', 'w')
	seed = 7
	np.random.seed(seed)

	inputs, outputs = pull_data(impute_inputs=False, impute_outputs=False, impute_method=1)
	bin_data(outputs[:,1]) # bin the continuous # STEM courses 

	encoded_inputs = None
	input_shape = inputs.shape
	for i in range(input_shape[1]):
		encoded_col = binarize_variable(inputs[:,i])
		if(encoded_inputs == None):
			encoded_inputs = encoded_col
		else:
			encoded_inputs = np.append(encoded_inputs, encoded_col, axis=1)

	encoded_outputs = None
	output_shape = outputs.shape
	for i in range(output_shape[1]):
		encoded_col = None
		if(i > 0 or i < 3):
			encoded_col = binarize_variable(outputs[:,i], ordinal=True)
		if(encoded_outputs == None):
			encoded_outputs = encoded_col
		else:
			encoded_outputs = np.append(encoded_outputs, encoded_col, axis=1)

	f_out.write("encoded_in\tencoded_out\tloss_fn\toptimizer\tactivation_hidden\tactivation_output\taccuracy\tstd_dev\n")
	for a in[False, True]:
		for b in[False, True]:
			for c in ['binary_crossentropy','categorical_crossentropy','mean_squared_error','mean_absolute_error']:
				for d in ['rmsprop','adam','sgd','adadelta','nadam']:
					for e in ['sigmoid','tanh','relu','hard_sigmoid','softmax']:
						for f in ['sigmoid','tanh','relu','hard_sigmoid','softmax']:
							(acc, sd) = (-1, -1)
							copied_inputs = inputs
							copied_outputs = outputs
							if(a):
								copied_inputs = encoded_inputs
							if(b):
								copied_outputs = encoded_outputs
							(acc, sd) = nn_run(copied_inputs, copied_outputs, use_mlpclass=True, no_hidden_layers=1, hidden_layer_size_multiplier=1, optimizer_fn=d, loss_fn=c, hidden_activation_fn=e, output_activation_fn=f)
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
else:
	seed = 7
	np.random.seed(seed)

	f_out = open('keras_out_structure.txt', 'w')
	f_out.write("input_inputs\timpute_outputs\timpute_method\tcross_validate\thidden_layers\thidden_layer_size\twhich_data\taccuracy\truns\tstd_dev\n")
	for a in [False, True]:
		for b in [False, True]:
			for c in [1, 2]:
				for d in [False]:
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
									# encoded_inputs = inputs
									# encoded_outputs = outputs
									input_shape = inputs.shape
									for i in range(input_shape[1]):
										encoded_col = binarize_variable(inputs[:,i])
										if(encoded_inputs == None):
											encoded_inputs = encoded_col
										else:
											encoded_inputs = np.append(encoded_inputs, encoded_col, axis=1)
									output_shape = outputs.shape
									for i in range(output_shape[1]):
										encoded_col = None
										if(i > 0 or i < 3):
											encoded_col = binarize_variable(outputs[:,i], ordinal=True)
										if(encoded_outputs == None):
											encoded_outputs = encoded_col
										else:
											encoded_outputs = np.append(encoded_outputs, encoded_col, axis=1)
								# skippable cases: imputing input or output while using test data
								if((a or b) and (g < 3)):
									(acc, sd) = (-1, -1)
								else:
									(acc, sd) = nn_run(encoded_inputs, encoded_outputs, use_mlpclass=d, no_hidden_layers=1, hidden_layer_size_multiplier=1, optimizer_fn='sgd', loss_fn='mean_squared_error', hidden_activation_fn='sigmoid', output_activation_fn='softmax')
								f_out.write(str(a) + "\t")
								f_out.write(str(b) + "\t")
								f_out.write(str(c) + "\t")
								f_out.write(str(d) + "\t")
								f_out.write(str(e) + "\t")
								f_out.write(str(f) + "\t")
								f_out.write(str(g) + "\t")
								if(d == False):
									# only one run, no std dev recorded
									f_out.write("%0.3f\t1\tNA\n" % acc)
								else:
									f_out.write("%0.3f\t10\t%.3f\n" % (acc, sd))










