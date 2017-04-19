# classifier.py
# trains and tests three classifiers: naive bayes, random forests and linear regression

import numpy as np
np.random.seed(111)

import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.linear_model

import sys

from preliminary_analysis import read_csv

# split the dataset into training and test sets
# negatives and positives are the negative instances in the dataset and the positive instances
# training_positives is the fraction of the positive examples to put in the training set
# ratio is the ratio of positive to negative examples to put in the training set
# the test set will always be 50% positive, 50% negative. This isn't really representative 
# (the test set should be indicative of the real-world ratio), fix if possible
def split_dataset(negatives, positives, training_positives, ratio):
	half_training_set = int(len(positives) * training_positives)
	half_test_set = int(len(positives) - half_training_set)
	training_set_length = half_training_set * (1 + ratio)
	test_set_length = half_test_set * 2
	train = np.append(positives[:half_training_set], negatives[:half_training_set * ratio],axis=0)
	np.random.shuffle(train)
	# EDIT: now I keep the test set balanced. We'll see how that affects things
	test = np.append(positives[half_training_set:half_training_set+half_test_set], 
		negatives[half_training_set * ratio: half_training_set * ratio + half_test_set],axis=0)
	np.random.shuffle(test)
	table = np.append(train,test, axis=0)

	# convert to one-hot coding using sklearn
	enc = sklearn.preprocessing.OneHotEncoder()
	table_onehot = enc.fit_transform(table[:,:-1])

	# split into training and test sets, sklearn handles validation on its own
	train = (table_onehot[:training_set_length], table[:training_set_length, -1])
	test = (table_onehot[training_set_length:training_set_length + test_set_length], 
		table[training_set_length:training_set_length + test_set_length, -1])
	return train, test

# train a model on the training set, then test it on the test set, printing results to the terminal
# model_name should be a string, for printing purposes only
# returns accuracy value
def train_and_test(model, model_name, training_set, test_set):

	model.fit(train[0], train[1])

	predictions = model.predict(test[0])
	print(model_name)
	c_m = sklearn.metrics.confusion_matrix(test[1], predictions)
	print(c_m)
	acc = sklearn.metrics.accuracy_score(test[1], predictions)
	print("Accuracy: ", acc)
	return acc


# parse csv
f = open(sys.argv[1])
table,labels,ordinal = read_csv(f)
l = len(table)

# split the dataset into positive and negative instances.
isnegative = np.array([x == '0' for x in table[:,-1]])
negatives = table[isnegative]
positives = table[isnegative == False]
np.random.shuffle(negatives)

max_accuracy = (0,'None')
# the dataset is imbalanced, try different amounts of training data with different ratios of positive to negative
# try different fractions of the positive examples in the training set vs. test set
for amount_of_positives in np.arange(.5,1,.1):
	print("Using ", amount_of_positives, " of the positive examples in the dataset:")
	# try different ratios of negative examples to positive examples in the training set
	for ratio in range(1,4):
		print("With ", ratio, " times as many negative examples:")
		
		train, test = split_dataset(negatives,positives,amount_of_positives,ratio)
		# random forest
		forest = sklearn.ensemble.RandomForestClassifier(min_samples_split=10)
		acc = train_and_test(forest, "Random Forest", train, test)
		if acc > max_accuracy[0]: max_accuracy = (acc, "Random Forest with " 
					+ str(amount_of_positives) + ' of the positive examples, using ' 
					+ str(ratio) + ' times as many negative examples')
		# naive bayes
		nb = sklearn.naive_bayes.BernoulliNB(binarize=None)
		acc = train_and_test(nb, "Naive Bayes", train, test)
		if acc > max_accuracy[0]: max_accuracy = (acc, "Naive Bayes with " 
					+ str(amount_of_positives) + ' of the positive examples, using ' 
					+ str(ratio) + ' times as many negative examples')

		# logistic regression
		reg = sklearn.linear_model.LogisticRegression()
		acc = train_and_test(reg, "Logistic Regression", train, test)
		if acc > max_accuracy[0]: max_accuracy = (acc, "Logistic Regression with " 
					+ str(amount_of_positives) + ' of the positive examples, using ' 
					+ str(ratio) + ' times as many negative examples')
print('Max accuracy is achieved by ', max_accuracy)