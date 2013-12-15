'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp
import cPickle as cp

prep = pp()

csv = open(r'..\data\train.csv', 'r')
training_data = open(r'..\data\training_data_small', 'w')

prep.import_csv(csv, training_data)
  
for key in prep.ids[:1]:
    print
    print "TEST - From class instance:"
    print prep.feature_dict[key]
    print prep.target_dict[key]

training_data = open(r'..\data\training_data_small', 'r')
data = cp.load(training_data)
print "TEST - From file:"
print data[0][1]
print data[1][1]