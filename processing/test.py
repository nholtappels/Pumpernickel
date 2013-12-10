'''
Created on 14.11.2013

@author: Nick
'''

from preprocessing.csvPreprocess import csvPreprocess as pp

if __name__ == '__main__':
    f = r"..\data\train.csv"
    test = pp()
    test.preprocess(f)
    for x in test.ids:
    	print test.feature_dict[x][0]

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
				