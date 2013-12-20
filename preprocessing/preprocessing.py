'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp

# look at KAPPA
# implement upper threshold
# association measure

lower_threshold = 5
upper_threshold = 100
numlines = 100
train = 1

# create dynamic filenames based on the values above

if train: csv = 'train.csv'
else: csv = 'test.csv'
storage = ('data_' + str(lower_threshold) + '_' +
           str(upper_threshold) + '_' + str(numlines))
features_csv = ('features_' + str(lower_threshold) + '_' +
                str(upper_threshold) + '_' + str(numlines) + '.csv')

targets_csv = ('targets_' + str(lower_threshold) + '_' +
               str(upper_threshold) + '_' + str(numlines) + '.csv')

# instanciate a model
prep = pp(lower_threshold, upper_threshold, numlines, train)

prep.import_csv(csv, storage)
prep.create_new_csvs(features_csv, targets_csv, storage)
