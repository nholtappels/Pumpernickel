pickle = open(r'..\data\training_data_small', 'r+')

data = load_data(pickle)
features = data[0]
targets = data[1]
ids = data[2]

# FEATURES: [[15,0,0,0,0,1,0,1,1,3,1,0,0,2,1,0,1,1,0,0,0,0,0,1],[...],...]; order: [id (not a feature), freq1, freq2, freq3, ..., has_hashtag, has_mention, is_retweet, oklahoma, new york, california...]
features_all_tweets = []
# TARGETS: [[0,1,0,0.194,0,0.605,0.2,0],...] probabilities for each kind of weather, not summing to 1. first value is ID, followed by target-values
targets_all_tweets = []




# build list of all words:
total_frequency = {}
all_words = []  # all used words
for tweet_id, features_old in features.items():
    tweet_words = features_old[0]
    for w in tweet_words:
        if not w in all_words:
            all_words.append(w)
            total_frequency[w] = 1
        else:
            total_frequency[w] += 1

    # e.g. for '#cold' also add 'cold' (and later increase frequency)
    # (hashtags have value in itself (more important) but should also increase frequency of word without #
    # because used as normal words (e.g. 'it's #freezing in #NYC today')
    for h in hashtags:
        if not w in all_words:
            all_words.append(w)
            total_frequency[w] = 1
        else:
            total_frequency[w] += 1

# throw out words occuring less that 'lower_threshold_word_frequencies':
new = []
for w in all_words:
    if not total_frequency[w] < lower_threshold_word_frequency:
        new.append(w)
all_words = new


# for later we need a list of states:
states = ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', \
        'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii', 'Idaho', \
        'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky', 'Louisiana', 'Maine', \
        'Maryland', 'Massachusetts', 'Michigan', 'Minnesota', 'Mississippi', \
        'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire', 'New Jersey', \
        'New Mexico', 'New York', 'North Carolina', 'North Dakota', 'Ohio', 'Oklahoma', \
        'Oregon', 'Pennsylvania', 'Rhode Island', 'South Carolina', 'South Dakota', \
        'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', \
        'American Samoa', 'District of Columbia', 'Guam', 'Northern Mariana Islands', \
        'Puerto Rico', 'United States Virgin Islands', 'West Virginia', \
        'Wisconsin', 'Wyoming']
new = []
for s in states:
    new.append(s.lower())
states = new



# for each tweet preprocess the features (stoplist, stemming etc.):
for tweet_id, features_old in features.items():
    features_new = []
    # ID:
    features_new.append(tweet_id)

    tweet_words = features_old[0]
    hashtags = features_old[1]
    has_hashtags = features_old[2]
    has_mention = features_old[3]
    is_retweeted = features_old[4]
    state = features_old[5]

    # WORDS:
    word_vector = [0] * len(all_words)
    for w in tweet_words:
        if w in all_words:  # not all words are in all_words because of threshold
            word_vector[all_words.index(w)] += 1

    # HASHTAGS:
    # e.g. for '#cold' also increase frequency of 'cold'
    for h in hashtags:
        if h in all_words:
            word_vector[all_words.index(h)] += 1

    # put word-features in list of features:
    for w in word_vector:
        features_new.append(w)



    # BINARY FEATURES:
    # make all values numbers, nicer for naive bayes
    features_new.append('1' if has_hashtags else '0')
    features_new.append('1' if has_mention else '0')
    features_new.append('1' if is_retweeted else '0')

    # STATES:
    # give naive bayes a list where 1 means it's that state: [0,0,0,0,1,0,0,0]
    # better than just a number representing a state because then gaussian naive
    # bayes would think that two neighbouring numbers are similar in some sense
    state_vector = [0] * len(states)
    if state in states:
        state_vector[states.index(state)] = 1
    for s in state_vector:
        features_new.append(s)




    # append features of this tweet to list of all features
    features_all_tweets.append(features_new)





# TARGETS:
for t_id, t in targets.items():
    # append this to targets: [ID, t1, t2, ...]
    targets_all_tweets.append([t_id] + t)






# WRITE TO FILES:
feat_file = open('../data/features.csv', 'w')


columns = ['ID']
for w in all_words:
    columns.append(w)
columns.append('has_hashtag')
columns.append('has_mention')
columns.append('is_retweeted')
for s in states:
    columns.append(s)

write_csv_row(feat_file, columns)

for row in features_all_tweets:
    write_csv_row(feat_file, row)

targ_file = open('../data/targets.csv', 'w')
writer = csv.writer(targ_file)
writer.writerows(targets_all_tweets)
