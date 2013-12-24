'''
Created on 15.12.2013

@author: Nick & David
'''

from csvPreprocess import csvPreprocess as pp
from create_filenames import create_names

lower_threshold = 5         #words occuring less often than this will be cut out
upper_threshold = 100       #words occuring more often than this as a percentage will be cut out
numlines_train = 100       #number of tweets used from training data
numlines_test = 100  # 0 will be interpreted as all lines       #number of tweets test data
only_trainingset = 1        #only preprocess the training set, will be split later on

def preprocess(lower_threshold, numlines_train, numlines_test, only_trainingset):
    csv_train, storage_train, features_train, targets_train, csv_test, \
    storage_test, features_test = create_names(lower_threshold, upper_threshold,
                                               numlines_train, numlines_test,
                                               only_trainingset, 1)
    '''preprocesses the training data and optionally a separate test data file. Creates features
    and extracts targets.
    '''
    
    prep = pp(lower_threshold, upper_threshold, numlines_train, numlines_test,
              only_trainingset)
    
    #import data from csv files:
    prep.import_csv(csv_train, csv_test, storage_train, storage_test)
    #create features and save to new csv files:
    prep.create_new_csvs(features_train, targets_train, features_test,
                         storage_train, storage_test)

if __name__ == '__main__':
    preprocess(lower_threshold, numlines_train, numlines_test, only_trainingset)
