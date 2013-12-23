'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp
from create_filenames import create_names

lower_threshold = 1
upper_threshold = 100
numlines_train = 10
numlines_test = 10  # 0 will be interpreted as all lines
only_trainingset = 0

def preprocess(lower_threshold, numlines_train, numlines_test, only_trainingset):
    csv_train, storage_train, features_train, targets_train, csv_test, \
    storage_test, features_test = create_names(lower_threshold, upper_threshold,
                                               numlines_train, numlines_test,
                                               only_trainingset, 1)

    # instantiate a model
    prep = pp(lower_threshold, upper_threshold, numlines_train, numlines_test,
              only_trainingset)

    prep.import_csv(csv_train, csv_test, storage_train, storage_test)
    prep.create_new_csvs(features_train, targets_train, features_test,
                         storage_train, storage_test)

if __name__ == '__main__':
    preprocess(lower_threshold, numlines_train, numlines_test, only_trainingset)
