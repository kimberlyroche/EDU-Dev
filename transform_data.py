import numpy as np
import sqlite3
import random

random.seed()

def pull_data(impute_inputs=False, impute_outputs=False):
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
	return (np.asarray(inputs), np.asarray(outputs))


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

# impute from mean
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

# impute more elaborately
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
	# for m in [0,1,2]:
		print "rows missing " + str(m) + " elements"
		if(m == 0):
			continue
		count = 0
		for i in range(len(v)):
			missing = []
			for j in range(len(v[i])):
				if(v[i][j] == null_label):
					missing.append(j)
			if(len(missing) == m):
				if(m == m_of_interest and verbose):
					print v[i]
				match_idx_list = find_matches(v, v[i], missing, null_label)
				if(len(match_idx_list) == 0):
					for k in missing:
						v[i][k] = mean_bycols[k]
					if(m == m_of_interest and verbose):
						print "> no matches, impute by column mean"
						print "> " + str(v[i])
				else:
					for k in match_idx_list:
						if(verbose):
							print "> matched: " + str(v[k])
					for k in missing:
						rand_idx = random.randint(0, len(match_idx_list)-1)
						v[i][k] = v[match_idx_list[rand_idx]][k]
						if(m == m_of_interest and verbose):
							print "> pick #" + str(rand_idx) + " at random to replace column " + str(k)
							print "> " + str(v[match_idx_list[rand_idx]])
					if(verbose):
						print "> FINAL " + str(v[i])
				if(m == m_of_interest and verbose):
					print "\n",

def binarize_variable(v, upper_range, ordinal=False):
	binarized = np.zeros(upper_range)
	for i in range(v):
		if(ordinal):
			binarized[i] = 1
		elif(i == v-1):
			binarized[i] = 1
	return binarized

inputs, outputs = pull_data(impute_inputs=False, impute_outputs=True)

print str(len(inputs)) + "x" + str(len(inputs[0]))
# impute2(inputs, 0)
# impute2(outputs, -1)

# input_range = 4
# binarized_input = np.empty(shape=(len(inputs), len(inputs[0])*input_range))
# for r in range(10):
# 	for c in range(len(inputs[r])):
# 		binarized_input[r][(c*input_range):((c+1)*input_range)] = binarize_variable(inputs[r][c], 4)

# np.savetxt('test_input.txt', inputs, fmt='%i', delimiter='\t')
# np.savetxt('test_output.txt', outputs, fmt='%i', delimiter='\t')

print outputs[:10]
print max(outputs[][])
# binarize_variable(outputs[0][1], 5, True)


