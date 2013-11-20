'''
Created on 14.11.2013

@author: Nick
'''

import sklearn
import csv
import numpy as np
from nltk.corpus import stopwords


lines= [line for line in file(r"..\data\train.csv")]

texts = []
words = {}
for line in lines[1:]:
	row = line.split(',')
	text = row[1].strip('"')

	text = text.split(' ')
	for w in text:
		if w.isalpha():
			w = w.lower()
			words.setdefault(w,0)
			words[w] += 1

#print words
print len(words)


