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
    when called with is_train == True this class only preprocesses the training set and splits it up in test set and training set. However, when called with is_train == False it trains on the training set and uses the same bag of words to predict on the test set given in a separate file. The second option is needed for the actual submission to Kaggle.
    '''
    def __init__(self, lower_threshold = 1, upper_threshold = 100, numlines_train = 0, numlines_test = 0, is_train = True):
        print "starting to preprocess..."
        self.features_dict_train = {}
        self.features_dict_test = {}
        self.targets_dict_train = {}

        # FEATURES: [[15,0,0,0,0,1,0,1,1,3,1,0,0,2,1,0,1,1,0,0,0,0,0,1],
        # [...],...]; order: [id (not a feature), freq1, freq2, freq3,
        # ..., has_hashtag, has_mention, is_retweet, has_link, oklahoma,
        # new york, california...]
        self.features_list_train = []
        self.features_list_test = []

        # TARGETS: [[0,1,0,0.194,0,0.605,0.2,0],. probabilities for each
        # kind of weather, not summing to 1. first value is ID, followed by
        # target-values
        self.targets_list_train = []

        self.all_words = []
        self.total_word_count = 0
        # The minimum number of occurrences of a word to be kept in all_words
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.upper_threshold_absolute = 0
        # If is_train == False, no target_dict will be created
        self.is_train = is_train
        # Number of lines of the initial CSV that will be imported
        self.numlines_train = numlines_train
        self.numlines_test = numlines_test
        self.delchars = (''.join(c for c in map(chr, range(256)) if not
                                 c.isalpha()))
        self.stopwords = corpus.stopwords.words('english')

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

    def import_csv(self, csv_train, csv_test, storage_train, storage_test):
        ''' Imports the csv containing the training data and optionally a csv containing the
        test data and creates two dictionaries from the training data (and optionally a feature_dict from
        the test data):

        feature_dict = {id1: features=[words=['w1', 'w2', etc.],
                        hashtags=['ht1', 'ht2', etc.],
                        has_hashtag=True/False, has_mention=True/False,
                        is_retweet=True/False, has_link=True/False state='state'], id2: etc.}

        target_dict = {id1: targets=[0, 0.2, 0.1, etc.], id2: etc.}
        '''

        # TRAINING SET FILE:
        print "importing training set..."
        source = open(csv_train, 'r')
        lines = [line for line in source]

        if self.numlines_train != 0:
            check_lines = lines[1:self.numlines_train]
        else:
            check_lines = lines[1:]

        i = 0
        for line in check_lines:
            # inform about progress
            if i % 500 == 0:
                print '%d tweets processed' % i
            i += 1

            # Process the line and create separate items from csv formatting
            line = line.strip('"\n')
            line = line.split('","')
            key = int(line[0])
            tweet = line[1]
            state = line[2]
            kinds = line[13:]
            # print kinds
            # Create features, targets and IDs and add them to the class variables
            features_train = self.__create_features_row(tweet, state)
            targets_train = self.__create_targets(kinds)
            self.features_dict_train[key] = features_train
            self.targets_dict_train[key] = targets_train
        print 'all tweets for training processed'

        # TEST SET FILE (if applicable):
        if self.is_train == False:
            source = open(csv_test, 'r')
            lines = [line for line in source]
            print "importing test set..."

            if self.numlines_test != 0:
                check_lines = lines[1:self.numlines_test]
            else:
                check_lines = lines[1:]

            i = 0
            for line in check_lines:
                # inform about progress
                if i % 500 == 0:
                    print '%d tweets processed' % i
                i += 1

                # Process the line and create separate items from csv formatting
                line = line.strip('"\n')
                line = line.split('","')
                key = int(line[0])
                tweet = line[1]
                state = line[2]
                # Create features and add them to the class variable
                features_test = self.__create_features_row(tweet, state)
                self.features_dict_test[key] = features_test
        print "all tweets for testing processed"
        self.__save_data(storage_train, storage_test)
        print "Data imported from CSVs"

    def create_new_csvs(self, features_train_csv, targets_train_csv,
                        features_test_csv, storage_train, storage_test):
        # load data from file into features_list and targets_list
        self.__load_data(storage_train, storage_test)
        self.total_frequency = self.__count_frequencies(self.features_dict_train)
        # optional: plotting of word frequencies
#         self.__plot_word_frequencies()
        self.__remove_lowfrequent_highfrequent_words()
        self.features_list_train = self.__create_features_list(self.features_dict_train)
        self.targets_list_train = self.__create_targets_list(self.targets_dict_train)

        # for test step we do not need to create the bag of words but just count word frequencies
        # according to the bag of words created for the training seet:
        if self.is_train == False:
            self.features_list_test = self.__create_features_list(self.features_dict_test)
        self.__save_csvs(features_train_csv, targets_train_csv, features_test_csv)

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
        # draw the lower and upper threshold:
        plt.axvline(x = self.lower_threshold, color = 'r')
        plt.axvline(x = self.upper_threshold_absolute, color = 'r')

        plt.xlabel('frequencies')
        plt.ylabel('count')
        plt.title('word frequencies')

        plt.show()

    def __create_features_row(self, tweet, state):
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
            if word.startswith('@'):
                has_mention = 1
                word = ''
            if (word.startswith('{link}') or word.startswith('http://') or
                word.startswith('https://')):
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

    def __save_data(self, storage_train, storage_test):
        print "saving data to storage file..."
        data = [self.features_dict_train, self.targets_dict_train]
        f = open(storage_train, 'w')
        cp.dump(data, f)
        f.close()
        # if using separate test set, also store it in separate storage:
        if self.is_train == False:
            data = [self.features_dict_test]
            f = open(storage_test , 'w')
            cp.dump(data, f)
            f.close()

    def __load_data(self, storage_train, storage_test):
        print "loading data from storage file..."
        source = open(storage_train , 'r')
        data = cp.load(source)
        self.features_dict_train = data[0]
        self.targets_dict_train = data[1]
        if self.is_train == False:
            source = open(storage_test , 'r')
            data = cp.load(source)
            self.features_dict_test = data[0]
        source.close()

    def __write_csv_row(self, f, row):
        for a in row[:-1]:
            f.write(str(a) + ',')
        f.write(str(row[-1]) + '\n')

    def __count_frequencies(self, features_dict):
        ''' Count the frequency of all words in the feature_dict
        increase it +1 for normal words and +2 for hashtags
        '''
        print "Counting frequencies..."
        total_frequency = {}
        for value in features_dict.values():
            for word in value[0]:
                total_frequency[word] = (total_frequency.setdefault(word, 0) + 1)
            # e.g. for '#cold' also add 'cold' (and later increase frequency)
            # (hashtags have value in itself (more important) but should also
            # increase frequency of word without # because used as normal
            # words (e.g. 'it's #freezing in #NYC today')
            for hashtag in value[1]:
                total_frequency[hashtag] = (total_frequency.setdefault(hashtag, 0) + 1)
        return total_frequency

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

        print "words less frequent than %d and more frequent than %d removed, \n %d/%d words kept" \
                % (self.lower_threshold, self.upper_threshold_absolute, self.total_word_count, num_all_words)

    def __create_features_list(self, features_dict):
        print "creating features list..."
        features_list = []
        for tweet_id, features_old in features_dict.iteritems():
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
            features_list.append(features_new)
        print "features list created."
        return features_list

    def __create_targets_list(self, targets_dict):
        print "Creating targets list..."
        targets_list = []
        for t_id, t in targets_dict.iteritems():
            # append this to targets: [ID, t1, t2, ...]
            targets_list.append([t_id] + t)

        return targets_list

    def __save_csvs(self, features_train_csv, targets_train_csv, features_test_csv):
        print "saving training features in csv"
        # first row: label of columns
        feature_headers = ['ID']
        for w in self.all_words:
            feature_headers.append(w)
        feature_headers.append('has_hashtag')
        feature_headers.append('has_mention')
        feature_headers.append('is_retweet')
        feature_headers.append('has_link')
        for s in self.states:
            feature_headers.append(s)

        # Save features_train in csv:
        features = open(features_train_csv , 'w')
        self.__write_csv_row(features, feature_headers)
        for row in self.features_list_train:
            self.__write_csv_row(features, row)
        features.close()

        target_headers = ['ID', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                          'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13',
                          'k14', 'k15']

        print "saving training targets in csv"
        # Save targets_train in csv:
        targets = open(targets_train_csv, 'w')
        self.__write_csv_row(targets, target_headers)
        for row in self.targets_list_train:
            self.__write_csv_row(targets, row)
        targets.close()

        # Save features_test in csv:
        if self.is_train == False:
            print "saving test features in csv"
            features = open(features_test_csv , 'w')
            self.__write_csv_row(features, feature_headers)
            for row in self.features_list_test:
                self.__write_csv_row(features, row)
            features.close()

            target_headers = ['ID', 'k1', 'k2', 'k3', 'k4', 'k5', 'k6',
                              'k7', 'k8', 'k9', 'k10', 'k11', 'k12', 'k13',
                              'k14', 'k15']
