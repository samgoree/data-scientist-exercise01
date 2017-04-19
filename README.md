## RTI CDS Analytics Exercise 01

This repository contains my submission for Exercise 01 as part of my application for Entry Level Data Scientist at RTI. My sqlite data extraction is in sqlite_to_csv.py, my preliminary analysis code is in preliminary_analysis.py, my feature selection code is in feature_selection.py and my classifier implementation is in classifier.py.

My analysis report document was written in LaTeX and is in RTI_analysis.pdf, source is in RIT_analysis.tex.

Update 4/19/17: Rearranging directory structure and cleaning up code for the sake of maintainability

 * source code is now in src folder
 * data and graphs (not pushed) have their own folders
 * moved csv parsing into a function and made sure it catches newline characters
 * reworked the creation of training and test sets to have a test set balanced between positive and negative examples. Given the unrealistic demographics on other attributes, I would want to do more research before concluding that the balance of negative to positive values in the dataset is not shifted from the real world.