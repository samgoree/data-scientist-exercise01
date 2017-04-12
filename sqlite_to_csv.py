# sqlite_to_csv
# Sam Goree
# accepts the sqlite database from RTI CDS Analytics Exercise 01 
# as command line argument writes to a csv file with the same name
# upsettingly not solved for databases in general, but so it goes with sql

import sqlite3
import sys

conn = sqlite3.connect(sys.argv[1])
c = conn.cursor()

f = open(sys.argv[1].split('.sqlite')[0] + '.csv', 'w')
f2 = open(sys.argv[1].split('.sqlite')[0] + 'numerical.csv', 'w')
print('File written to ' + f.name + ', ' + f2.name)

f.write("age,workclass,education_level,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_week,country,over_50k\n")
f.write("1,0,0,0,0,0,0,0,0,1,1,1,0,0\n") # label for my use whether each category is nominal (0) or ordinal (1)
f2.write("age,workclass,education_level,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_week,country,over_50k\n")
f2.write("1,0,0,0,0,0,0,0,0,1,1,1,0,0\n") # label for my use whether each category is nominal (0) or ordinal (1)


# I'm having a busy week, I found a way to extract the table I want, so I'm going with it even if it looks ugly
# I don't know sql well enough to do this sort of thing more elegantly
for row in c.execute(
	"""SELECT records.age, workclasses.name, education_levels.name, 
	records.education_num, marital_statuses.name, occupations.name, relationships.name,
	races.name, sexes.name, records.capital_gain, records.capital_loss, records.hours_week,
	countries.name, records.over_50k
	FROM ((((((((records 
		INNER JOIN workclasses on records.workclass_id = workclasses.id)
		INNER JOIN education_levels on records.education_level_id = education_levels.id)
		INNER JOIN marital_statuses on records.marital_status_id = marital_statuses.id)
		INNER JOIN occupations on records.occupation_id = occupations.id)
		INNER JOIN relationships on records.relationship_id = relationships.id)
		INNER JOIN races on records.race_id = races.id)
		INNER JOIN sexes on records.sex_id = sexes.id)
		INNER JOIN countries on records.country_id = countries.id)"""):
	for field in row:
		f.write(str(field) + ',')
	f.write('\n')
for row in c.execute(
	"""SELECT * FROM 'records'"""):
	for field in row[1:]:
		f2.write(str(field) + ',')
	f2.write('\n')