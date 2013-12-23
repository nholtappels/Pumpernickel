'''
Created on 16.12.2013

@author: Nick
'''

from NaiveBayes import NaiveBayes
from create_filenames import create_names
import numpy as np
from slice_merge import slice_csv, merge_csvs

lower_threshold = 1
upper_threshold = 100
numlines_train = 10  # 0 will be interpreted as all lines
numlines_test = 10000  # 0 will be interpreted as all lines
slice_size = 200

features_file_train, targets_file_train, features_file_test, \
predictions_file_test = create_names(lower_threshold, upper_threshold,
                                     numlines_train, numlines_test, 0, 0)

def main():

    try:
        load_features(features_file_train, 1)
    except IOError:
        print "The corresponding files have not been created yet."
        print "Please run preprocessing witht the same parameters and try again."
        raise SystemExit(0)

    slice_names = slice_csv(features_file_test, slice_size)


    # TRAINING DATA:
    print 'load training features...'
    features_train, dont_use_me = load_features(features_file_train, 1)
    print 'load training targets...'
    all_targets_train = load_targets(targets_file_train)

    s = 1

    pred_names = []
    first = 1

    print 'TRAINING'
    nb = [None] * 15
    all_targets = all_targets_train.T
    for i in xrange(15):
        nb[i] = NaiveBayes()
        targets = all_targets[i]
        print 'target %d: training' % (i)
        nb[i].train(features_train, targets)

    print 'PREDICTING'
    for slice_name in slice_names:
        print '>>> starting with slice', s
        print 'load test features...'
        # TEST DATA:
        features_test, IDs_test = load_features(slice_name, 0)

        # PREDICTION:
        print 'start predicting...'
        predictions_file_test = slice_name.replace('features', 'predictions')
        pred_names.append(predictions_file_test)
        predictions_file = open(predictions_file_test, 'w')
        if first:
            write_csv_row(predictions_file, ['id', 's1', 's2', 's3', 's4', 's5', 'w1', 'w2', 'w3',
                                             'w4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8',
                                             'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15'])
            first = 0


        # run Naive Bayes for each target separately
        # (not 1 vs all because different targets are independent)

        all_predictions = []
        for i in xrange(15):
            print 'target %d: predicting' % (i)
            predictions = nb[i].predict(features_test)
            all_predictions.append(predictions)

        print 'write predictions to file...'
        # write predictions to file
        all_predictions = (np.array(all_predictions).T).tolist()
        print all_predictions
        for i in range(len(all_predictions)):
            print i
            pred = all_predictions[i]
            ID = IDs_test[i]
            zeros = [int(ID)] + [0] * 9
            write_csv_row(predictions_file, zeros + pred)

        print '>>> finished with slice', s
        s += 1
        predictions_file.close()
        print
    print '>>> >>> finished with all slices <<<'
    merge_csvs(pred_names)
    return

def write_csv_row(f, row):
    for a in row[:-1]:
        f.write(str(a) + ',')
    f.write(str(row[-1]) + '\n')

def load_features(features_file, skip):
    # determine number of columns in order to skip the first column
    if skip == 1:
        features = np.loadtxt(features_file, delimiter = ',', skiprows = 1)
    if skip == 0:
        features = np.loadtxt(features_file, delimiter = ',')
    IDs = features[:, 0]
    features = features[:, 1:]  # first column is ID
    return features, IDs

def load_targets(targets_file):
    targets = np.loadtxt(targets_file, delimiter = ',', skiprows = 1)
    targets = targets[:, 1:]  # first column is ID
    return targets

def evaluate(predictions, correct):
    errors = 0
    for i in range(0, len(predictions)):
        if(predictions[i] != correct[i]):
            errors += 1
    return errors

def splitdata(data, targets, IDs):
    # split in training set & test set:
    cut = .7 * len(data)
    testdata = data[cut:]
    testtargets = targets[cut:]
    testIDs = IDs[cut:]
    data = data[:cut]
    targets = targets[:cut]
    IDs = IDs[:cut]

    return data, targets, IDs, testdata, testtargets, testIDs

if __name__ == '__main__':
    main()
