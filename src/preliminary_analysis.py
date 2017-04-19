# preliminary_analysis.py
# takes a csv file as an argument and outputs a histogram or bar graph of each variable
# conducts chi-square tests of independence between each pair of attributes 
# (numerical attributes are split into quartiles)

import sys

import numpy as np
import scipy.stats
import matplotlib.pyplot
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 10})

# makes a matplotlib histogram from a single numerical attribute
def create_histogram(table_column, attribute_name):
	plt.hist(table_column, 25)
	plt.xlabel(attribute_name)
	plt.ylabel('Frequency')
	plt.title(attribute_name + ' Distribution')

# Attach a text label above each bar displaying its height
def autolabel(rects, ax):
	for rect in rects:
		height = rect.get_height()
		ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
				'%d' % int(height),
				ha='center', va='bottom')

# makes a matplotlib barchart from a single nominal attribute
def create_barchart(table_column, attribute_name):
	frequencies = {}
	for val in table_column:
		if val not in frequencies.keys():
			frequencies[val] = 0
		else:
			frequencies[val] += 1
	# fix the order of categories -- python dictionaries do not preserve order and keys() may be inconsistent
	categories = sorted(frequencies.keys())
	ax = plt.subplot(111)
	rects = ax.bar(np.arange(len(categories)), [frequencies[c] for c in categories], width=1/2)
	autolabel(rects, ax)
	ax.set_xticks(np.arange(len(categories)) + 1/4)
	ax.set_xticklabels(categories)
	ax.set_title(attribute_name)

# create a contingency table between two attributes
# column1 and column2 are the data vectors
# attribute1_ordinal and attribute2_ordinal are boolean values to indicate that the attributes are numerical
# ncategories1 and ncategories2 is the number of partitions to make for numerical attributes
# 		If the attributes are nominal, these can be none, default is 4 if numerical and unspecified
def columns_to_contingency_table(column1, attribute1_ordinal, column2, 
			attribute2_ordinal, ncategories1=None,ncategories2=None):
	# calculate quartiles - this is inefficient (O(N) space), but takes more code to do well, refactor later
	if attribute1_ordinal == 1:
		if ncategories1 is None: ncategories1=4
		low = min(column1.astype('int32'))
		high = max(column1.astype('int32'))
		quartile = (high-low)/ncategories1
		column1values = np.zeros(len(column1), dtype='int32')
		for i, val in enumerate(column1.astype('int32')):
			column1values[i] = int(val//quartile)
		valuedict1 = dict([(i,i) for i in range(ncategories1)])
		valuedict1[ncategories1] = ncategories1-1
	else: # otherwise, just list out all the values
		column1values = column1
		i = 0
		valuedict1 = {}
		for val in column1values:
			if val not in valuedict1:
				valuedict1[val] = i # valuedict maps values to rows/columns in the contingency table
				i+=1
		if ncategories1 is None: ncategories1 = i
	# same as above
	if attribute2_ordinal == 1:
		if ncategories2 is None: ncategories2=4
		low = min(column2.astype('int32'))
		high = max(column2.astype('int32'))
		quartile = (high-low)/ncategories2
		column2values = np.zeros(len(column2), dtype='int32')
		for i, val in enumerate(column2.astype('int32')):
			column2values[i] = int(val//quartile)
		valuedict2 = dict([(i,i) for i in range(ncategories2)])
		valuedict2[ncategories2] = ncategories2-1
	else: 
		column2values = column2
		i = 0
		valuedict2 = {}
		for val in column2values:
			if val not in valuedict2:
				valuedict2[val] = i
				i+=1
		if ncategories2 is None: ncategories2 = i
	# create the table
	table = np.zeros([ncategories1, ncategories2])
	for val1,val2 in zip(column1values,column2values):
		table[valuedict1[val1],valuedict2[val2]] += 1
	return table

# parse a csv file, f, into a numpy array
# returns a numpy array of the data, two python lists of length shape[1] of the array with 
# labels and ordinal or not booleans respectively
def read_csv(f):
	table_builder = []
	for i,line in enumerate(f):
		line = line.split(',\n')[0]
		if i == 0:
			labels = line.split(',')
		elif i == 1:
			# the last element may be a newline
			ordinal = [int(i) for i in line.split(',')] 
		else:
			table_builder.append(line.split(','))
	# convert to a numpy array
	return np.array(table_builder), labels, ordinal


if __name__=='__main__':	
	# this is not for general use, no need to use argparse or anything of that nature
	if len(sys.argv[1]) == 0 or sys.argv[1] == '-h':
		print("Call with the path to exercise01.sqlite as the argument")

	# parse csv, this is a relatively small dataset so we don't have to worry about memory
	f = open(sys.argv[1])
	table,labels,ordinal = read_csv(f)
	# make and display charts
	for i,(l,o) in enumerate(zip(labels, ordinal)):
		if o == 1:
			create_histogram(table[:,i].astype('int32'), l)
			plt.show()
		else:
			create_barchart(table[:,i], l)
			plt.show()

	#make an interesting graph, as per the prompt
	data = table[:,8].astype('object') + ',' + table[:,6].astype('object')
	create_barchart(data, 'relationship vs. sex')
	plt.show()
	# conduct chi square analysis on each pair of attributes, remember all of the pairs with p < 0.05
	# This did not work, and in hindsight was a mistake to waste time on
	# but I am including the code for transparency
	"""
	significant_pairs = []
	for i in range(table.shape[1]):
		for j in range(i+1,table.shape[1]):
			contingency_table = columns_to_contingency_table(table[:,i], ordinal[i], table[:,j], ordinal[j])
			print(contingency_table)
			if (contingency_table < 5).sum() == 0:
				_, p, _, _ = scipy.stats.chi2_contingency(contingency_table)
				if p < 0.05: significant_pairs.append((labels[i],labels[j],p))
			else:
				print('Pair', (labels[i],labels[j]), 'has insufficient data for correlation')
	print(significant_pairs)"""