'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp

prep = pp(3, 500)

csv = 'train.csv'
storage = 'training_data_5000_3'
features_csv = 'features_5000_3.csv'
targets_csv = 'targets_5000_3.csv'

prep.import_csv(csv, storage)
prep.create_new_csvs(features_csv, targets_csv, storage)