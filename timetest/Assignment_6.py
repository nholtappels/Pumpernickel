'''
Created on 22.11.2013

@author: Nick
'''

from gaussian_naive_bayes import GaussianNaiveBayes as GNB
#from gaussian_naive_bayes_2 import GaussianNaiveBayes as GNB_2
import numpy as np
import timeit
import time



# def error_rate(predictions, targets):
#         v = targets - predictions
#         errors = 0
#         for e in v:
#             if e[0] != 0:
#                 errors += 1
#         
#         return errors, (float(errors) / predictions.shape[0])
 
houses = np.loadtxt("RealEstate.csv", delimiter=",",
                        skiprows=1,usecols=(2,3,4,5))

def exercise_1_1(houses):
    ''' 1.1 Test on housing data
    '''
    
    
#     # randomize the order of the datapoints in iris
#     np.random.shuffle(houses)
    
    # separate targets from houses
    targets = houses[:, 0]
    
    threshold = 295000
    
    new_targets = []
    for x in targets:
        if x >= threshold:
            new_targets.append(1)
        else:
            new_targets.append(0)
    targets = np.array(new_targets)
    
    total_length = targets.size
    targets = targets.reshape((total_length, 1))
    
    # separate data from houses
    data = houses[:,1:]
    
    split = .7
    
    # separate test data & targets from the rest
    train_data = data[:total_length * split, :]
    train_targets = targets[:total_length * split, :]
    
    test_data = data[total_length * split:, :]
    test_targets = targets[total_length * split:, :]
    
#     print "Testing on housing data"
#     print "Budget threshold: " + str(threshold / 1000) + "K $"
#     print "Split: " + str(int(split * 100)) + "/" + str(int((1 - split) * 100))
    
    model = GNB()
    model.train(train_data, train_targets)
    
    predictions = model.test(test_data)
    
#     error = error_rate(predictions, test_targets)
    
#     print "Total error: " + str(int(error[0]))
#     print "Error rate: " + str(int(round(error[1], 2) * 100)) + "%"
#     print
    pass
# 
# def exercise_1_2():
#     ''' 1.2 Test on Iris data
#     '''
#     
#     iris = np.loadtxt("..\..\Resources\data\iris.csv",delimiter=",",
#                          skiprows=1)
#     
#     # randomize the order of the datapoints in iris
#     np.random.shuffle(iris)
#     
#     # separate targets and data from iris
#     data = iris[:, :-1]
#     targets = iris[:, -1]
#     
#     total_length = targets.size
#     targets = targets.reshape((total_length, 1))
#     
#     split = .7
#     
#     # separate test data & targets from the rest
#     train_data = data[:total_length * split, :]
#     train_targets = targets[:total_length * split, :]
#     
#     test_data = data[total_length * split:, :]
#     test_targets = targets[total_length * split:, :]
#     
#     print "Testing on Iris data"
#     print "Split: " + str(int(split * 100)) + "/" + str(int((1 - split) * 100))
#     
#     model = GNB()
#     model.train(train_data, train_targets)
#     
#     predictions = model.test(test_data)
#     
#     error = error_rate(predictions, test_targets)
#     
#     print "Total error: " + str(int(error[0]))
#     print "Error rate: " + str(int(round(error[1], 2) * 100)) + "%"
#     print
#     pass
# 
# def exercise_2():
#     houses = np.loadtxt("..\..\Resources\data\RealEstate.csv", delimiter=",",
#                         skiprows=1,usecols=(2,3,4,5))
#     
#     # randomize the order of the datapoints in iris
#     np.random.shuffle(houses)
#     
#     # separate targets from houses
#     targets = houses[:, 0]
#     
#     threshold = 295000
#     
#     new_targets = []
#     for x in targets:
#         if x >= threshold:
#             new_targets.append(1)
#         else:
#             new_targets.append(0)
#     targets = np.array(new_targets)
#     
#     total_length = targets.size
#     targets = targets.reshape((total_length, 1))
#     
#     
#     
#     # separate data from houses
#     data = houses[:,1:]
#         
#     # separate test data & targets from the rest
#     train_data = data[:-10, :]
#     train_targets = targets[:-10, :]
#     
#     test_data = data[-10:, :]
#     test_targets = targets[-10:, :]
#     
#     # duplicate above budget in test_data and test_targets
#     train_data_b = train_data
#     train_targets_b = train_targets
#     
#     for i in xrange(train_targets.size):
#         if train_targets[i][0] == 1:
#             train_data_b = np.vstack([train_data_b, train_data[i]])
#             train_targets_b = np.vstack([train_targets_b, train_targets[i]])
#     
#     print "Testing on housing data A"
#     print "Budget threshold: " + str(threshold / 1000) + "K $"
#     
#     model = GNB()
#     model.train(train_data, train_targets)
#     
#     predictions = model.test(test_data)
#     
#     error = error_rate(predictions, test_targets)
#     
#     print "Total error: " + str(int(error[0]))
#     print "Error rate: " + str(int(round(error[1], 2) * 100)) + "%"
#     print
#     
#     print "Testing on housing data B"
#     print "Budget threshold: " + str(threshold / 1000) + "K $"
#     
#     model = GNB_2()
#     model.train(train_data_b, train_targets_b)
#     
#     predictions = model.test(test_data)
#     
#     error_2 = error_rate(predictions, test_targets)
#     
#     print "Total error: " + str(int(error_2[0]))
#     print "Error rate: " + str(int(round(error_2[1], 2) * 100)) + "%"
#     print
#     print "Identical error on data A and data B:", (error[0] == error_2[0])
#     pass

# timer = timeit.Timer(exercise_1_1(houses))
# print timer.timeit()
# # exercise_1_2()
# # exercise_2()
times = []
for x in xrange(10):
    # randomize the order of the datapoints in iris
    np.random.shuffle(houses)
    time_1 = time.time()
    exercise_1_1(houses)
    time_2 = time.time()
    times.append((time_2 - time_1))

print np.mean(times)
