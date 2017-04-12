# classifier.py
# runs three classifiers: naive bayes, random forests and linear regression

import numpy as np
import sklearn.ensemble
import sklearn.preprocessing
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.linear_model

import sys


# parse csv, this is a relatively small dataset so we don't have to worry about memory
f = open(sys.argv[1])
table_builder = []
for i,line in enumerate(f):
	if i == 0:
		labels = line.split(',')[:-1]
	elif i == 1:
		categorical = [int(val)==0 for val in line.split(',')[:-1]]
	else:
		table_builder.append(line.split(',')[:-1]) # the last element has a newline character after the last comma
# convert to a numpy array
table = np.array(table_builder, dtype='int32')
l = len(table)

# balance the training set so that there is less skew towards "<50k"
isnegative = table[:,-1] == 0
negatives = table[isnegative]
positives = table[isnegative == False]
np.random.shuffle(negatives)

max_accuracy = (0,'None')
# the dataset is imbalanced, try different amounts of training data with different ratios of positive to negative
# try different fractions of the positive examples in the training set
for amount_of_positives in np.arange(.5,1,.1):
	print("Using ", amount_of_positives, " of the positive examples in the dataset:")
	# try different ratios of negative examples to positive examples
	for ratio in range(1,5):
		print("With ", ratio, " times as many negative examples:")
		half_training_set = int(len(positives) * amount_of_positives)
		training_set_length = half_training_set * 2
		train = np.append(positives[:half_training_set], negatives[:half_training_set * ratio],axis=0)
		np.random.shuffle(train)
		test = np.append(positives[half_training_set:], negatives[half_training_set * ratio:],axis=0)
		np.random.shuffle(test)
		table = np.append(train,test, axis=0)

		# convert to one-hot coding using sklearn
		enc = sklearn.preprocessing.OneHotEncoder()
		table_onehot = enc.fit_transform(table[:,:-1])

		# split into training and test sets, sklearn handles validation on its own
		train = (table_onehot[:training_set_length], table[:training_set_length, -1])
		test = (table_onehot[training_set_length:], table[training_set_length:, -1])

		# random forest results
		forest = sklearn.ensemble.RandomForestClassifier(min_samples_split=10)
		forest.fit(train[0], train[1])

		predictions = forest.predict(test[0])
		print("Random Forest")
		c_m = sklearn.metrics.confusion_matrix(test[1], predictions)
		print(c_m)
		acc = sklearn.metrics.accuracy_score(test[1], predictions)
		print("Accuracy: ", acc)
		if acc > max_accuracy[0]: max_accuracy = (acc, "Random Forest " + str(amount_of_positives) + ',' + str(ratio))

		# naive bayes
		nb = sklearn.naive_bayes.BernoulliNB(binarize=None)
		nb.fit(train[0], train[1])
		predictions = nb.predict(test[0])
		print("Naive Bayes")
		c_m = sklearn.metrics.confusion_matrix(test[1], predictions)
		print(c_m)
		acc = sklearn.metrics.accuracy_score(test[1], predictions)
		print("Accuracy: ", acc)
		if acc > max_accuracy[0]: max_accuracy = (acc, "Naive Bayes " + str(amount_of_positives) + ',' + str(ratio))


		# logistic regression
		reg = sklearn.linear_model.LogisticRegression()
		reg.fit(train[0], train[1])
		predictions = reg.predict(test[0])
		print("Logistic Regression")
		print(test[1])
		c_m = sklearn.metrics.confusion_matrix(test[1], predictions)
		print(c_m)
		acc = sklearn.metrics.accuracy_score(test[1], predictions)
		print("Accuracy: ", acc)
		if acc > max_accuracy[0]: max_accuracy = (acc, "Logistic Regression " + str(amount_of_positives) + ',' + str(ratio))
print(max_accuracy)