'''
Created on 14.12.2013

@author: Nick
'''

from convert_files import *

pickle = open(r'..\data\new_train', 'r+')

data = load_data(pickle)
features = data[0]
targets = data[1]
ids = data[2]



print features[1]