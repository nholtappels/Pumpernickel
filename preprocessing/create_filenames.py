'''
Created on 23.12.2013

@author: Nick
'''

def create_names(lower_threshold, upper_threshold, numlines_train, numlines_test, only_trainingset, ispre):
    ''' if ispre: output for preprocessing, if not ispre: output for processing
    '''
    # dummy values in case not used:
    csv_test = ''
    storage_test = ''
    features_test = ''

    # create dynamic filenames based on the values above
    csv_train = '../data/train.csv'
    storage_train = ('../data/_data_train_' + str(lower_threshold) + '_' +
           str(upper_threshold) + '_' + str(numlines_train))
    features_train = ('../data/_features_train_' + str(lower_threshold) + '_' +
                str(upper_threshold) + '_' + str(numlines_train) + '.csv')
    targets_train = ('../data/_targets_train_' + str(lower_threshold) + '_' +
               str(upper_threshold) + '_' + str(numlines_train) + '.csv')

    if only_trainingset == False:
        csv_test = '../data/test.csv'
        storage_test = ('../data/_data_test_' + str(lower_threshold) + '_' +
               str(upper_threshold) + '_' + str(numlines_test))
        features_test = ('../data/_features_test_' + str(lower_threshold) + '_' +
                    str(upper_threshold) + '_' + str(numlines_test) + '.csv')

    predictions_test = ('../data/_predictions_test_' + str(lower_threshold) + '_' +
                    str(upper_threshold) + '_' + str(numlines_test) + '.csv')

    if ispre:
        return csv_train, storage_train, features_train, targets_train, csv_test, storage_test, features_test
    else:
        return features_train, targets_train, features_test, predictions_test
