'''
Created on 16.12.2013

@author: Nick
'''

from NaiveBayes import NaiveBayes
from create_filenames import create_names
from preprocessing import preprocess
import numpy as np

lower_threshold = 10
upper_threshold = 100
numlines_train = 10000  # 0 will be interpreted as all lines
numlines_test = 1000  # 0 will be interpreted as all lines

features_file_train, targets_file_train, features_file_test, \
predictions_file_test = create_names(lower_threshold, upper_threshold,
                                numlines_train, numlines_test, 0, 0)

def main():

    try:
        load_features(features_file_train)
    except IOError:
        preprocess(lower_threshold, numlines_train, numlines_test, 0)

    nb = NaiveBayes()

    # TRAINING DATA:
    print 'load training features...'
    features_train = load_features(features_file_train)
    print 'load training targets...'
    all_targets_train = load_targets(targets_file_train)

    # TEST DATA:
#    print 'split data in training set and test set...'
#    features_train, all_targets_train, IDs_train, features_test,
#    all_targets_test, IDs_test = splitdata(features, all_targets, IDs)
    print 'load test features...'
    features_test, IDs_test = load_features(features_file_test)

    # PREDICTION:
    print 'start predicting...'
    predictions_file = open(predictions_file_test, 'w')
    write_csv_row(predictions_file, ['id', 's1', 's2', 's3', 's4', 's5', 'w1', 'w2', 'w3', \
                                    'w4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', \
                                    'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15'])


    # run Naive Bayes for each target separately
    # (not 1 vs all because different targets are independent)

    # print 'test on:'
    # print testfeatures
    all_predictions = []
    all_targets = all_targets_train.T
    i = 0
    for targets in all_targets:
    # targets = all_targets[3]
    # print 'targets: ' + str(targets)
        print 'target %d: training and predicting' % (i)
        # Naive Bayes:
        nb.train(features_train, targets)
        predictions = nb.predict(features_test)
        # errors = evaluate(predictions, testtargets)
        # print 'target %d: %d/%d predictions wrong.' %(targ_index,errors, len(predictions))
#         print '--> predictions: ' + str(predictions) + '\n'
        all_predictions.append(predictions)
        i += 1

    print 'write predictions to file...'
    # write predictions to file
    all_predictions = (np.array(all_predictions).T).tolist()
    for i in range(len(all_predictions)):
        pred = all_predictions[i]
        ID = IDs_test[i]
        zeros = [int(ID)] + [0] * 9
        # zeros = [0.0] * 10
        write_csv_row(predictions_file, zeros + pred)

    print 'finished.'
    return

def write_csv_row(f, row):
    for a in row[:-1]:
        f.write(str(a) + ',')
    f.write(str(row[-1]) + '\n')

def load_features(features_file):
    # determine number of columns in order to skip the first column
    #features = np.loadtxt(features_file, delimiter = ',', skiprows = 1)
    #IDs = features[:, 0]
    #features = features[:, 1:]  # first column is ID
    #return features, IDs
    
    f = open(features_file)
    reader = csv.reader(f)
    firstline = reader.next()   # first line are column labels
    IDs = []
    features = []
    for line in reader:
        IDs.append(int(line[0]))
        features.append([int(a) for a in line[1:]])
    return np.asarray(features), np.asarray(IDs)



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
