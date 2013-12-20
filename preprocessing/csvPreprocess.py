'''
Created on 08.12.2013

@author: Nick
'''

from nltk import corpus
import cPickle as cp
import matplotlib.pyplot as plt
import operator

class csvPreprocess(object):
    ''' Class for preprocessing csv file for later use
    '''

    '''
    parameters:
    lower_threshold: words occuring less frequent than this are removed (absolute)
    upper_threshold: words occuring more frequent than this are removed (threshold is percentage of highest frequency, 100=nothing removed)

    '''
    def __init__(self, lower_threshold = 1, upper_threshold = 100, numlines = 0, is_train = True):
        self.feature_dict = {}
        self.target_dict = {}
        self.all_words = []
        self.total_word_count = 0
        # The minimum number of occurrences of a word to be kept in all_words
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.upper_threshold_absolute = 0
        # If is_train == False, no target_dict will be created
        self.is_train = is_train
        # Number of lines of the initial CSV that will be imported
        self.numlines = numlines
        self.delchars = (''.join(c for c in map(chr, range(256)) if not
                                 c.isalpha()))
        self.stopwords = corpus.stopwords.words('english')
        # self.stopwords.append('weather')   #this word occurs in almost 50% of the tweets
        # FEATURES: [[15,0,0,0,0,1,0,1,1,3,1,0,0,2,1,0,1,1,0,0,0,0,0,1],
        # [...],...]; order: [id (not a feature), freq1, freq2, freq3,
        # ..., has_hashtag, has_mention, is_retweet, oklahoma, new york,
        # california...]
        self.feature_list = []
        # TARGETS: [[0,1,0,0.194,0,0.605,0.2,0],...] probabilities for each
        # kind of weather, not summing to 1. first value is ID, followed by
        # target-values
        self.target_list = []
        self.total_frequency = {}
        # for later we need a list of states
        self.states = ['alabama', 'alaska', 'arizona', 'arkansas',
                       'california', 'colorado', 'connecticut', 'delaware',
                       'florida', 'georgia', 'hawaii', 'idaho', 'illinois',
                       'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana',
                       'maine', 'maryland', 'massachusetts', 'michigan',
                       'minnesota', 'mississippi', 'missouri', 'montana',
                       'nebraska', 'nevada', 'new hampshire', 'new jersey',
                       'new mexico', 'new york', 'north carolina',
                       'north dakota', 'ohio', 'oklahoma', 'oregon',
                       'pennsylvania', 'rhode island', 'south carolina',
                       'south dakota', 'tennessee', 'texas', 'utah',
                       'vermont', 'virginia', 'washington',
                       'american samoa', 'district of columbia', 'guam',
                       'northern mariana islands', 'puerto rico',
                       'united states virgin islands', 'west virginia',
                       'wisconsin', 'wyoming']

    def import_csv(self, csv, storage):
        ''' Imports the csv containing the training data and creates two
        dictionaries from it:

        feature_dict = {id1: features=[words=['w1', 'w2', etc.],
                        hashtags=['ht1', 'ht2', etc.],
                        has_hashtag=True/False, has_mention=True/False,
                        is_retweet=True/False, has_link=True/False state='state'], id2: etc.}

        target_dict = {id1: targets=[0, 0.2, 0.1, etc.], id2: etc.}
        '''

        source = open(r'..\data\\' + csv , 'r')
        lines = [line for line in source]
        print "CSV imported"

        if self.numlines != 0:
            check_lines = lines[1:self.numlines]
        else:
            check_lines = lines[1:]

        for line in check_lines:
            # Process the line and create separate items from csv formatting
            line = line.strip('"\n')
            line = line.split('","')
            key = int(line[0])
            tweet = line[1]
            state = line[2]
            kinds = line[13:]
            # print kinds
            # Create features, targets and IDs and add them to the class params
            features = self.__create_features(tweet, state)
            targets = self.__create_targets(kinds)
            self.feature_dict[key] = features
            if self.is_train:
                self.target_dict[key] = targets
        print "Data imported from CSV"
        self.__save_data(storage)

    def create_new_csvs(self, features_csv, targets_csv, storage):
        self.__load_data(storage)
        self.__count_frequencies()
        self.__plot_word_frequencies()
        self.__remove_lowfrequent_highfrequent_words()
        self.__create_feature_list()
        self.__create_target_list()
        self.__save_csvs(features_csv, targets_csv)

    def __plot_word_frequencies(self):
        count_freq_dict = {}
        for freq in self.total_frequency.itervalues():
            count_freq_dict.setdefault(freq, 0)
            count_freq_dict[freq] += 1
        freq_list = []
        count_list = []
        for freq, count in count_freq_dict.iteritems():
            # print '%d %d' % (freq, count)
            freq_list.append(freq)
            count_list.append(count)

        plt.bar(freq_list, count_list)
        plt.axvline(x = self.lower_threshold, color = 'r')
        plt.axvline(x = self.upper_threshold_absolute, color = 'r')


        plt.xlabel('frequencies')
        plt.ylabel('count')
        plt.title('word frequencies')

        plt.show()

    def __create_features(self, tweet, state):
        ''' Creates one row of the feature_dict based on the content of
        the csv. Is called from within import_csv.
        '''


        words = []
        hashtags = []
        has_hashtag = 0
        has_mention = 0
        is_retweet = 0
        has_link = 0
        # Process the words
        tweet = tweet.split(' ')
        for word in tweet:
            word = word.lower()
            if word.startswith('@mention'):
                has_mention = 1
                word = ''
            if word.startswith('{link}'):
                has_link = 1
                word = ''
            if word.startswith('#'):
                hash_word = word.translate(None, self.delchars)
                hashtags.append(hash_word.lower())
                has_hashtag = 1
            word = word.translate(None, self.delchars)
            if word in self.stopwords:
                word = ''
            if len(word) > 1:
                words.append(word)

        if has_mention:
            if 'rt' in words:
                is_retweet = 1
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

    def __save_data(self, storage):
        target = open(r'..\data\\' + storage , 'w')
        data = [self.feature_dict, self.target_dict]
        cp.dump(data, target)
        target.close()
        print "Data saved"

    def __load_data(self, storage):
        source = open(r'..\data\\' + storage , 'r')
        data = cp.load(source)
        self.feature_dict = data[0]
        self.target_dict = data[1]
        source.close()
        print "Data loaded"

    def __write_csv_row(self, f, row):
        for a in row[:-1]:
            f.write(str(a) + ',')
        f.write(str(row[-1]) + '\n')

    def __count_frequencies(self):
        ''' Count the frequency of all words in the feature_dict
        increase it +1 for normal words and +2 for hashtags
        '''
        for value in self.feature_dict.values():
            for word in value[0]:
                self.total_frequency[word] = (self.total_frequency.
                                              setdefault(word, 0) + 1)
            # e.g. for '#cold' also add 'cold' (and later increase frequency)
            # (hashtags have value in itself (more important) but should also
            # increase frequency of word without # because used as normal
            # words (e.g. 'it's #freezing in #NYC today')
            for hashtag in value[1]:
                self.total_frequency[hashtag] = (self.total_frequency.
                                                 setdefault(hashtag, 0) + 1)
        print "Frequencies counted"

    def __remove_lowfrequent_highfrequent_words(self):
        ''' Throw out words occuring less often than 'lower_threshold' and
        more often than 'upper_threshold'
        Also count the total number of words in the dict and save it
        to self.word_count and create list of all words
        '''
        # make upper_threshold absolute using the percentage of maximum frequency
        maxfreq = max(self.total_frequency.iteritems(), key = operator.itemgetter(1))[1]
        self.upper_threshold_absolute = maxfreq * self.upper_threshold / 100
        num_all_words = len(self.total_frequency)
        for word, freq in self.total_frequency.items():
            if freq < self.lower_threshold or freq > self.upper_threshold_absolute:
                del self.total_frequency[word]
            # if freq > 200:
            #    print '%s:\t\t%d' % (word, freq)
            self.total_word_count = len(self.total_frequency)
        for word in self.total_frequency.iterkeys():
            self.all_words.append(word)

        print "words less frequent than %d and more frequent than %d removed, %d/%d words kept" % (self.lower_threshold, self.upper_threshold_absolute, self.total_word_count, num_all_words)

    def __create_feature_list(self):
        for tweet_id, features_old in self.feature_dict.iteritems():
            features_new = []

            # ID:
            features_new.append(tweet_id)

            words = features_old[0]
            hashtags = features_old[1]
            has_hashtags = features_old[2]
            has_mention = features_old[3]
            is_retweet = features_old[4]
            has_link = features_old[5]
            state = features_old[6]

            # WORDS:
            word_vector = [0] * self.total_word_count
            for w in words:
                if w in self.all_words:
                    word_vector[self.all_words.index(w)] += 1

            # HASHTAGS:
            # e.g. for '#cold' also increase frequency of 'cold'
            for h in hashtags:
                if h in self.all_words:
                    word_vector[self.all_words.index(h)] += 1

            # put word-features in list of features:
            for frequency in word_vector:
                features_new.append(frequency)

            # BINARY FEATURES:
            features_new.append(has_hashtags)
            features_new.append(has_mention)
            features_new.append(is_retweet)
            features_new.append(has_link)

            # STATES:
            # give naive bayes a list where 1 means it's that state:
            # [0,0,0,0,1,0,0,0]
            # better than just a number representing a state because then
            # gaussian naive bayes would think that two neighboring states
            # are related
            state_vector = [0] * len(self.states)
            if state in self.states:
                state_vector[self.states.index(state)] = 1
            for s in state_vector:
                features_new.append(s)
            # append features of this tweet to list of all features
            self.feature_list.append(features_new)

        print "Feature list created"

    def __create_target_list(self):
        for t_id, t in self.target_dict.iteritems():
            # append this to targets: [ID, t1, t2, ...]
            self.target_list.append([t_id] + t)

        print "Target list created"

    def __save_csvs(self, features_csv, targets_csv):

        feature_headers = ['ID']
        for w in self.all_words:
            feature_headers.append(w)
        feature_headers.append('has_hashtag')
        feature_headers.append('has_mention')
        feature_headers.append('is_retweet')
        feature_headers.append('has_link')
        for s in self.states:
            feature_headers.append(s)

        # Save features.csv
        features = open(r'..\data\\' + features_csv , 'w')
        self.__write_csv_row(features, feature_headers)
        for row in self.feature_list:
            self.__write_csv_row(features, row)
        features.close()
        print "Features CSV created"

        target_headers = ['ID', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                          'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13',
                          'k14', 'k15']

        # Save targets.csv
        targets = open(r'..\data\\' + targets_csv , 'w')
        self.__write_csv_row(targets, target_headers)
        for row in self.target_list:
            self.__write_csv_row(targets, row)
        targets.close()
        print "Targets CSV created"
