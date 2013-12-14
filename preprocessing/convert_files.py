'''
Created on 14.11.2013

@author: Nick
'''

from csvPreprocess import csvPreprocess as pp
import cPickle as cp

if __name__ == '__main__':
    csv = r'..\data\train.csv'
    pickle = open(r'..\data\new_train_2', 'w')

def convert_data(source, target):
    train_data = pp()
    train_data.preprocess(source)
    print "pre-processing done"
    dump_data = [train_data.feature_dict, train_data.target_dict, train_data.ids]
    cp.dump(dump_data, target)
    print "data converted"

def load_data(source):
    data = cp.load(source)
    return data

convert_data(csv, pickle)
    
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