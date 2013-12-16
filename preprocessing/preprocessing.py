'''
Created on 15.12.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp

prep = pp()

csv = 'train.csv'
storage = 'training_data_3'
features_csv = 'features_2.csv'
targets_csv = 'targets_2.csv'

prep.import_csv(csv, storage)
prep.create_new_csvs(features_csv, targets_csv, storage)