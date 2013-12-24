'''
Created on 16.12.2013

@author: Nick
'''

from NaiveBayes import NaiveBayes
from create_filenames import create_names
import numpy as np
from slice_merge import slice_csv, merge_csvs

lower_threshold = 5
upper_threshold = 100
numlines_train = 100  # 0 will be interpreted as all lines

split_at = 0.7


features_file_train, targets_file_train, features_file_test, \
probabilities_filename, predictions_filename = create_names(lower_threshold, upper_threshold,
                                     numlines_train, 0, 1, 0)

def main():
    features = None
    IDs = None
    all_targets = None
    try:
        print 'load training features...'
        features, IDs = load_features(features_file_train, 1)
        print 'load training targets...'
        all_targets = load_targets(targets_file_train)
    except IOError:
        print "The corresponding files have not been created yet."
        print "Please run preprocessing with the same parameters and try again."
        raise SystemExit(0)

    print 'split data...'
    features_train, all_targets_train, IDs_train, features_test, targets_test, IDs_test\
            = splitdata(features, all_targets, IDs)


    # run Naive Bayes for each target separately
    # (not 1 vs all because different targets are independent)
    all_targets_train = all_targets_train.T
    all_probabilities = []
    for i in xrange(15):
        print 'TARGET %d:' % (i)
        print 'train...'
        nb = NaiveBayes()
        targets = all_targets_train[i]

        nb.train(features_train, targets)

        # PREDICTION:
        print 'predict...'        
        probabilities = nb.predict(features_test)
        all_probabilities.append(probabilities)


    print 'write predictions to file...'
    predictions_file = open(predictions_filename, 'w')
    write_csv_row(predictions_file, ['id', 's1', 's2', 's3', 's4', 's5', 'w1', 'w2',\
                                     'w3','w4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',\
                                     'k7', 'k8','k9', 'k10', 'k11', 'k12', 'k13',\
                                     'k14', 'k15'])
    prob_file = open(probabilities_filename, 'w')
    write_csv_row(prob_file, ['id', 's1', 's2', 's3', 's4', 's5', 'w1', 'w2',\
                                     'w3','w4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',\
                                     'k7', 'k8','k9', 'k10', 'k11', 'k12', 'k13',\
                                     'k14', 'k15'])

    all_prob = (np.array(all_probabilities).T).tolist()
    all_predictions = []
    for i in range(len(all_prob)):
        prob = all_prob[i]
        ID = IDs_test[i]
        zeros = [int(ID)] + [0] * 9
        write_csv_row(prob_file, zeros + prob)
        #make predictions from probabilities (either 0 or 1):
        pred = [round(p) for p in prob]
        write_csv_row(predictions_file, zeros + pred)
        all_predictions.append(pred)
    prob_file.close()
    predictions_file.close()
    
    print ''
    print 'EVALUATE PREDICTIONS'
    row_errors = 0
    total_errors = 0
    number_tweets = len(targets_test)
    predictions_total = 15 * number_tweets
    for i in xrange(number_tweets):
        target_row = targets_test[i]
        targets_rounded = [round(p) for p in target_row]
        predictions = all_predictions[i]
        row_wrong = False
        for j in xrange(len(targets_rounded)):
            if targets_rounded[j] != predictions[j]:
                row_wrong = True
                total_errors += 1
        if row_wrong:
            row_errors += 1
    row_accuracy = (float(number_tweets - row_errors)/number_tweets) * 100
    total_accuracy = (float(predictions_total - total_errors))\
            / predictions_total* 100
    print '%d/%d tweets contain an error in the predictions \
            --> accuracy = %d percent' % (row_errors, number_tweets, row_accuracy)
    print '%d/%d predictions in total wrong --> accuracy %d percent.' \
                      % (total_errors, predictions_total, total_accuracy)
    

    print 'finished.'

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
