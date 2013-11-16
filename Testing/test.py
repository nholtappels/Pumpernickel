'''
Created on 14.11.2013

@author: Nick
'''

import sklearn
import csv
import numpy as np
from nltk.corpus import stopwords

f = open(r"..\..\ML\data\Project\train.csv")

f.readline()


model = sklearn.naive_bayes.MultinomialNB()

