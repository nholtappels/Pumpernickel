'''
Created on 08.12.2013

@author: Nick
'''

from nltk import corpus
import cPickle as cp

class csvPreprocess(object):
    ''' Class for preprocessing csv file for later use
    '''

    def __init__(self, is_train = True):
        self.feature_dict = {}
        self.target_dict = {}
        self.ids = []
        self.is_train = is_train
        self.delchars = ''.join(c for c in map(chr, range(256)) if not c.isalpha())
        self.stopwords = corpus.stopwords.words('english')
    
    def import_csv(self, source, target):
        ''' Imports the csv containing our training data and creates three
        items from it:
        
        feature_dict = {id1: features=[words=['w1', 'w2', etc.],
                        hashtags=['ht1', 'ht2', etc.],
                        has_hashtag=True/False, has_mention=True/False, 
                        is_retweet=True/False, state='state'], id2: etc.}
    
        target_dict = {id1: targets=[0, 0.2, 0.1, etc.], id2: etc.}
    
        ids = [id1, id2, etc.]
        '''
        
        lines = [line for line in source]
        print "CSV imported"
        
        for line in lines[1:]:    
            # Process the line and create separate items from csv formatting
            line = line.strip('"\n')
            line = line.split('","')
            key = int(line[0])
            tweet = line[1]
            state = line[2]
            kinds = line[13:]
            # Create features, targets and IDs and add them to the class params
            features = self.__create_features(tweet, state)
            targets = self.__create_targets(kinds)
            self.feature_dict[key] = features
            if self.is_train:
                self.target_dict[key] = targets
            self.ids.append(key)
        print "Data imported from CSV"   
        self.__save_data(target)
        
        
    def __create_features(self, tweet, state):
        ''' Creates one row of the feature_dict based on the content of
        the csv. Is called from within import_csv.
        '''
        
        words = []
        hashtags = []
        has_hashtag = False
        has_mention = False
        is_retweet = False
        has_link = False
        # Process the words
        tweet = tweet.split(' ')
        for word in tweet:
            if word.startswith('@mention'):
                has_mention = True
                word = ''
            if word.startswith('{link}'):
                has_link = True
                word = ''
            if word.startswith('#'):
                hash_word = word.translate(None, self.delchars)
                hashtags.append(hash_word.lower())
                has_hashtag = True
            word = word.translate(None, self.delchars)
            if word in self.stopwords:
                word = ''
            if len(word) > 1:
                words.append(word.lower())
            
        if has_mention:
            if 'rt' in words:
                is_retweet = True
                words.remove('rt')
        features = [words, hashtags, has_hashtag, has_mention, is_retweet, has_link, state]
        return features
        
    def __create_targets(self, kinds):
        ''' Creates one row of the target_dict based on the content of
        the csv. Is called from within import_csv.
        '''
        targets = []
        for t in kinds:
            targets.append(t)
        return targets
    
    def __save_data(self, target):
        data = [self.feature_dict, self.target_dict, self.ids]
        cp.dump(data, target)
        print "Data saved"

    def load_data(self, source):
        data = cp.load(source)
        self.feature_dict = data[0]
        self.target_dict = data[1]
        self.ids = data[2]
        print "Data loaded"