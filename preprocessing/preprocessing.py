'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp

prep = pp()
 
csv = 'train.csv'
storage = 'training_data_2'

# prep.import_csv(csv, storage)

prep.load_data(storage)

for key in prep.ids[:1]:
    print
    print "TEST - From class instance:"
    print prep.feature_dict[key]
    print prep.target_dict[key]