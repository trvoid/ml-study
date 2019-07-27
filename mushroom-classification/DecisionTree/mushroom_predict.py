import sys
import numpy as np
import pandas as pd
import treepredict

if len(sys.argv) == 1:
  print 'Usage: python', sys.argv[0], 'path-to-dataset-file'
  sys.exit(0)
  
################################################################################
#    Load data                                                                 #
################################################################################

print '\n------------------------'
print '\n  Load data             '
print '\n------------------------'

dataset_file = sys.argv[1]
data = pd.read_csv(dataset_file)
data.describe()

feature_columns = data.columns[1:]
for i, f in zip(np.arange(1, len(feature_columns) + 1), feature_columns):
    print 'feature {:d}:\t{}'.format(i, f)

# Relocate the class column from the first position to the last position.

my_data = data.as_matrix()

row_count, col_count = my_data.shape

class_col = np.copy(my_data[:, 0])
my_data[:, 0:col_count-1] = np.copy(my_data[:, 1:col_count])
my_data[:, col_count-1] = class_col

################################################################################
#    Build tree                                                                #
################################################################################

print '\n------------------------'
print '\n  Build tree            '
print '\n------------------------'

tree = treepredict.buildtree(my_data)

treepredict.drawtree(tree)
treepredict.printtree(tree)

################################################################################
#    Predict observations if they are poisonous or edible                      #
################################################################################

print '\n------------------------'
print '\n  Predict observations  '
print '\n------------------------'

observations = pd.read_csv('observations.csv').as_matrix()

for i in xrange(observations.shape[0]):
  print observations[i]
  results = treepredict.classify(observations[i], tree)
  print 'Prediction ==>', results
