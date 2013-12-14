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

pickle = open(r'..\data\new_train_test', 'r+')

data = load_data(pickle)
features = data[0]
targets = data[1]
ids = data[2]

stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('down')
stopwords.remove('very')

all_features = []
words = []
#for each tweet preprocess the features (stoplist, stemming etc.):
for key,value in features.items():
	features = []
	features.append(key)





<<<<<<< HEAD
#print features[1][0]
=======
print features[1]
>>>>>>> 033d559f23687dedef48dd33442254d17887a8b2
