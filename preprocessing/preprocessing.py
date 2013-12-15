'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as cp

prep = cp()

csv = open(r'..\data\train.csv', 'r')
training_data = open(r'..\data\training_data_2', 'w')

# prep.import_csv(csv, training_data)
prep.load_data(training_data)

for key in prep.ids[:3]:
    print prep.feature_dict[key]
    print prep.target_dict[key]