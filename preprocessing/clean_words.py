'''
Created on 14.12.2013

@author: Nick
'''

from nltk import corpus
from convert_files import load_data, dump_data

old = open(r'..\data\new_train', 'r')
new = open(r'..\data\new_train_3', 'w')

data = load_data(old)

features = data[0]
targets = data[1]
ids = data[2]
new_features = {}

delchars = ''.join(c for c in map(chr, range(256)) if not c.isalpha())
stopwords = corpus.stopwords.words('english')
stopwords.remove('up')
stopwords.remove('down')

for key in ids:
    words = features[key][0]
    hashtags = features[key][1]
    has_hashtag = features[key][2]
    has_mention = features[key][3]
    is_retweet = features[key][4]
    has_link = features[key][5]
    state = features[key][6]
    new_words = []
    for word in words:
        if word.startswith('@mention'):
            has_mention = True
            word = ''
        if word.startswith('{link}'):
            has_link = True
            word = ''
        if word.startswith('#'):
            hash_word = word.translate(None, delchars)
            hashtags.append(hash_word.lower())
            has_hashtag = True
        word = word.translate(None, delchars)
        if word in stopwords:
            word = ''
        if len(word) > 1:
            new_words.append(word.lower())
        
    if has_mention:
        if 'rt' in new_words:
            is_retweet = True
            new_words.remove('rt')
    
    new_features[id] = [new_words, hashtags, has_hashtag, has_mention, is_retweet, has_link, state]

new_data = [new_features, targets, ids]
dump_data(new_data, new)