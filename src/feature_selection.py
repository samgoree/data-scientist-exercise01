# feature_selection.py
# takes a csv file of numerical attributes (exercise01numerical.csv included here),
# removes education_num, modifies capital gain, loss and country to boolean values

import numpy as np
import sys
from preliminary_analysis import read_csv

f = open(sys.argv[1])
table,labels,ordinal = read_csv(f)

for i,l in enumerate(labels):
	# turn capital gain and loss into boolean values
	if l == 'capital_gain' or l == 'capital_loss':
		table[:,i] = (table[:,i] > 0).astype('int32')
	# change country to "is country US"
	if l == 'country':
		table[:,i] = (table[:,i] == 40).astype('int32')
	# remove education_num, it is similar to education level except noisier
	if l == 'education_num':
		table = np.delete(table,i,axis=1)
		del(labels[i])
		del(categorical[i])

# write a new file
f_out = open(sys.argv[1].split('.csv')[0] + 'selected.csv', 'w')
for label in labels:
	f_out.write(label + ',')
f_out.write('\n')
for c in categorical:
	f_out.write(('1' if c else '0') + ',')
f_out.write('\n')
for line in table:
	for val in line:
		f_out.write(str(val) + ',')
	f_out.write('\n')