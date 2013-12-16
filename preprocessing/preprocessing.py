'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp

# look at KAPPA
# implement upper threshold
# association measure

frequency_threshold = 15
numlines = 10000

prep = pp(frequency_threshold, numlines)

csv = 'train.csv'
storage = 'training_data_' + str(frequency_threshold) + '_' + str(numlines)
features_csv = 'features_' + str(frequency_threshold) + '_' + str(numlines) + '.csv'
targets_csv = 'targets_' + str(frequency_threshold) + '_' + str(numlines) + '.csv'

prep.import_csv(csv, storage)
prep.create_new_csvs(features_csv, targets_csv, storage)