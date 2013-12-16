'''
Created on 16.12.2013

@author: Nick
'''

from processing.NaiveBayes import NaiveBayes
import numpy as np

feature_filename = 
targets_filename = 

def main(features, targets):

    nb = NaiveBayes()

    # TWITTER:
    features, all_targets, IDs = load_features_targets(features, targets)

    # test,test_indeces = load_testdata('../test.csv')
    features, all_targets, IDs, testfeatures, all_testtargets, testIDs = splitdata(features,
            all_targets, IDs)


    prediction_file = open('../data/predictions.csv', 'w')

    write_csv_row(prediction_file, ['id', 's1', 's2', 's3', 's4', 's5', 'w1', 'w2', 'w3', \
                                    'w4', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'k7', 'k8', \
                                    'k9', 'k10', 'k11', 'k12', 'k13', 'k14', 'k15'])


    # run Naive Bayes for each target separately
    # (not 1 vs all because different targets are independent)

    # print 'test on:'
    # print testfeatures

    all_predictions = []
    all_targets = all_targets.T
    for targets in all_targets:
    # targets = all_targets[3]
    # print 'targets: ' + str(targets)
        # Naive Bayes:
        nb.train(features, targets)
        predictions = nb.predict(testfeatures)
        # errors = evaluate(predictions, testtargets)
        # print 'target %d: %d/%d predictions wrong.' %(targ_index,errors, len(predictions))
#         print '--> predictions: ' + str(predictions) + '\n'
        all_predictions.append(predictions)

    # write predictions to file
    all_predictions = (np.array(all_predictions).T).tolist()
    for i in range(len(all_predictions)):
        pred = all_predictions[i]
        # ID = testIDs[i]
        # zeros = [ID] + [0] * 9
        zeros = [0.0] * 10
        write_csv_row(prediction_file, zeros + pred)



    # nb.train(data, targets)
    # predictions = nb.predict(testdata)

    # errors = evaluate(predictions, testtargets)
    # print 'Housing Data: %d/%d predictions wrong.' %(errors, len(predictions))

    return

def write_csv_row(f, row):
    for a in row[:-1]:
        f.write(str(a) + ',')
    f.write(str(row[-1]) + '\n')

def load_features_targets(features, targets):
    # determine number of columns in order to skip the first column
    feature_file = '../data/' + str(features)
    target_file = '../data/' + str(targets)
    features = np.loadtxt(feature_file, delimiter = ',', skiprows = 1)
    features = features[:, 1:]  # first column is ID

    targets = np.loadtxt(target_file, delimiter = ',', skiprows = 1)
    targets = targets[:, 1:]  # first column is ID

    IDs = features[:0]

    return features, targets, IDs

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
    main(feature_filename, targets_filename)