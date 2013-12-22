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
numlines_train = 5000
numlines_test = 0   #0 will be interpreted as all lines
only_trainingset = False

#dummy values in case not used:
csv_test = ''
storage_test = ''
features_test = ''

# create dynamic filenames based on the values above
csv_train = '../data/train.csv'
storage_train = ('../data/data_train_' + str(lower_threshold) + '_' +
       str(upper_threshold) + '_' + str(numlines_train))
features_train = ('../data/features_train_' + str(lower_threshold) + '_' +
            str(upper_threshold) + '_' + str(numlines_train) + '.csv')
targets_train = ('../data/targets_train_' + str(lower_threshold) + '_' +
           str(upper_threshold) + '_' + str(numlines_train) + '.csv') 

if only_trainingset == False:
    csv_test = '../data/test.csv'
    storage_test = ('../data/data_test_' + str(lower_threshold) + '_' +
           str(upper_threshold) + '_' + str(numlines_test))
    features_test = ('../data/features_test_' + str(lower_threshold) + '_' +
                str(upper_threshold) + '_' + str(numlines_test) + '.csv')

# instantiate a model
prep = pp(lower_threshold, upper_threshold, numlines_train, numlines_test, only_trainingset)

prep.import_csv(csv_train, csv_test, storage_train, storage_test)
prep.create_new_csvs(features_train, targets_train, features_test, storage_train, storage_test)
