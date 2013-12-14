'''
Created on 14.12.2013

@author: Nick
'''
'''  
    data_f = {id1: features=[words=['w1', 'w2', etc.], hashtags=['ht1', 'ht2', etc.],
                        has_hashtag=True/False, has_mention=True/False, 
                        is_retweet=True/False, state='state'], id2: etc.}
    
    data_t = {id1: targets=[0, 0.2, 0.1, etc.], id2: etc.}
    
    ids = [id1, id2, etc.]
'''

from convert_files import *
import nltk


global lower_threshold_word_frequency
lower_threshold_word_frequency = 1	#if words occur less than threshold, don't include in all_words


pickle = open(r'..\data\new_train_2', 'r+')

data = load_data(pickle)
features = data[0]
targets = data[1]
ids = data[2]

stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('down')
stopwords.remove('very')

features_all_tweets = []


#build list of all words:
total_frequency = {}
all_words = []	#all used words
for tweet_id,features_old in features.items():
	tweet_words = features_old[0]
	for w in tweet_words:
		if not w in all_words:
			all_words.append(w)
			total_frequency[w] = 1
		else:
			total_frequency[w] += 1
		if '#' in w:
			print w
			#e.g. for '#cold' also add 'cold' (and later increase frequency)
	#(hashtags have value in itself (more important) but should also increase frequency of word without #
	#because used as normal words (e.g. 'it's #freezing in #NYC today')
			w = w.strip('#')
			print w
			if not w in all_words:
				all_words.append(w)
				total_frequency[w] = 1
			else:
				total_frequency[w] += 1

#throw out words occuring less that 'lower_threshold_word_frequencies':
for w in all_words:
	if total_frequency[w] < lower_threshold_word_frequency:
		all_words.remove(w)
	
print all_words

#for each tweet preprocess the features (stoplist, stemming etc.):
for tweet_id,features_old in features.items():
	features_new = []
	#ID:
	features_new.append(tweet_id)

	tweet_words = features_old[0]
	hashtags = features_old[1]
	has_hashtags = features_old[2]
	has_mention = features_old[3]
	is_retweeted = features_old[4]
	state = features_old[5]
			
	#WORDS:
	word_vector = [0]*len(all_words)
	for w in tweet_words:
		word_vector[all_words.index(w)] += 1

	#HASHTAGS:
	#e.g. for '#cold' also increase frequency of 'cold'
	for h in hashtags:
		word_vector[all_words.index(h.strip('#'))] += 1
	
	features_new.append(word_vector)



	#OTHER FEATURES:
	features_new.append(has_hashtags)
	features_new.append(has_mention)
	features_new.append(is_retweeted)
	features_new.append(state)

	print features_new
	print " "


#print features[1][0]
