'''
Created on 14.11.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp
import cPickle as cp

def covert_files():
    csv = r'..\data\train.csv'
    pickle = open(r'..\data\new_train', 'r+')
    train_data = pp()
    train_data.preprocess(csv)
    print "pre-processing done"
    dump_data = [train_data.feature_dict, train_data.target_dict, train_data.ids]
    dump_data(dump_data, pickle)
    print "data converted"

def dump_data(data, target):
    cp.dump(data, target)

def load_data(source):
    data = cp.load(source)
    return data
    
# def count_words():
# 	words = {}
# 	for line in lines[1:]:
# 		row = line.split(',')
# 		text = row[1].strip('"')
# 	
# 		text = text.split(' ')
# 		for w in text:
# 			if w.isalpha():
# 				w = w.lower()
# 				words.setdefault(w,0)
# 				words[w] += 1
# 	return len(words)
# 
# 	for line in lines[1:]:
# 		row = line.split(',')
# 		text = row[1].strip('"')
# 	
# 		text = text.split(' ')
# 		words = []
# 		for w in text:
# 			if w.isalpha():
# 				w = w.lower()
# 				words.append(w)