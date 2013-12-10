'''
Created on 10.12.2013

@author: Nick
'''

from bayesian.gaussian_naive_bayes import GaussianNaiveBayes as GNB
import numpy as np

def error_rate(predictions, targets):
        v = targets - predictions
        errors = 0
        for e in v:
            if e[0] != 0:
                errors += 1
        
        return errors, (float(errors) / predictions.shape[0])

def test1(samples):
    print "Now testing with " + str(samples) + " samples"
    all_data = np.loadtxt(r"..\data\train_2.csv", delimiter=";",
                        skiprows=1)
    
    np.random.shuffle(all_data)
        
    all_data = all_data[:samples]
    
    data = all_data[:, 1:18]
    
    for x in xrange(15):
    
        targets = all_data[:, 18+x]
        targets = targets.reshape((samples, 1))
            
        split = .7
        
        # separate test data & targets from the rest
        train_data = data[:samples * split, :]
        train_targets = targets[:samples * split, :]
        
        test_data = data[samples * split:, :]
        test_targets = targets[samples * split:, :]
    
        model = GNB()
        model.train(train_data, train_targets)
        
        predictions = model.test(test_data)
        
        error = error_rate(predictions, test_targets)
        
        print ("k" + str(x+1) + " - "+ "Total error: " + str(int(error[0]))
               + " - Error rate: " + str(int(round(error[1], 2) * 100)) + "%")

test1(10000)