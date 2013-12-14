'''
Created on 08.12.2013

@author: Nick
'''

class csvPreprocess(object):
    '''
    Class for preprocessing csv file for later use
    
    data_f = {id1: features=[words=['w1', 'w2', etc.], hashtags=['ht1', 'ht2', etc.],
                        has_hashtag=True/False, has_mention=True/False, 
                        is_retweet=True/False, state='state'], id2: etc.}
    
    data_t = {id1: targets=[0, 0.2, 0.1, etc.], id2: etc.}
    
    ids = [id1, id2, etc.]
    '''

    def __init__(self):
        self.feature_dict = {}
        self.target_dict = {}
        self.ids = []
    
    def preprocess(self, f):
    
        lines = [line for line in file(f)]
        
        for line in lines[1:]:    
            # Process the line and create separate items from csv formatting
            line = line.strip('"')
            line = line.split('","')
            key = int(line[0])
            tweet = line[1]
            state = line[2]
            kinds = line[13:]
            # Create features, targets and IDs and add them to the class params
            features = self.__create_features(tweet, state)
            targets = self.__create_targets(kinds)
            self.feature_dict[key] = features
            self.target_dict[key] = targets
            self.ids.append(key)
        
    def __create_features(self, tweet, state):
        features = []
        words = []
        hashtags = []
        has_hashtag = False
        has_mention = False
        is_retweet = False
        has_link = False
        # Process the words
        tweet = tweet.split(' ')
        for w in tweet:
            words.append(w)
        for x in [words, hashtags, has_hashtag, has_mention, is_retweet,
                  has_link, state]:
            features.append(x)
        return features
        
    def __create_targets(self, kinds):
        targets = []
        for t in kinds:
            targets.append(t)
        return targets