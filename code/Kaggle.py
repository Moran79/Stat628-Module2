# Import Essential Packages
import json
import sklearn
import numpy as np
import pandas as pd
import nltk
import re
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
import nltk.sentiment
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from pyitlib import discrete_random_variable as drv

##### 1. PREPROCESSING #####

### 1.1 STOPWORDS ###
stopwords = set(nltk.corpus.stopwords.words('english'))  # We choose some stopwords from library
keep1 = {"no", "nor", "not", "don", "don't", "shouldn't", "ain", "aren", "aren't", 'couldn', "couldn't", 'didn',
         'doesn',
         "doesn't", "didn't", "wouldn't", 'wouldn', "won't", "won", "shan't", "hasn't", "hadn't", 'hadn', 'hasn',
         'haven', "haven't",
         "weren't", 'weren', "wasn't", 'wasn', "very", "couldn't", 'isn', 'mightn', "mightn't", 'mustn', "mustn't",
         'needn', "needn't",
         "isn't", 'shan', "shan't", 'shouldn', "shouldn't", }  # Negative words
keep2 = {"all", "above", "below", "up", "down"}  # Something useful in stopwords
for word in keep1:
    stopwords.remove(word)
for word in keep2:
    stopwords.remove(word)
add1 = set("'ve")  # Add some stopwords
for word in add1:
    stopwords.add(word)


### 1.2 GOOD / BAD WORD ###
# Read in Pos & Neg dictionary by Minqing Hu and Bing Liu
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
good_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/positive-words.txt',
                            header=None,names=['word'])['word'])
bad_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/negative-words.txt',
                           encoding="ISO-8859-1", header=None, names=['word'])['word'])

# Add _NEG to the opposite dictionary
# Goodword + _NEG -> Badword
# Badword + _NEG -> Goodword
good_word, bad_word = good_word | set(x + '_NEG' for x in bad_word), bad_word | set(x + '_NEG' for x in good_word)


### 1.3 STEMMING ###
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


### 1.4 TOKENIZING ###
def tokenization_and_stemming_all(text):
    ### FEATURE: attitude ###
    # Measure the intensity of reviewer's emotion
    # There are two types of emotion I choose
    # 1. All CAP words also indicate strong attitude.
    # Use four as lower limit of the word length
    # 2. The occurance of several continous ? or ! also indicates a strong attitude
    attitude = 0  # Initialize reviewer's attitude
    attitude += len(re.findall(r'[A-Z]{4,}', text))
    attitude += len(re.findall(r'[?!]+', text)) # !? exists
    attitude += len(re.findall(r'[?!]{2,}', text)) # !? occurance >= 2

    ### TEXT CLEANING ###
    # nltk.word_tokenize would split [don't] to [do, n't]
    # But [n't] could not be recognized by nltk.sentiment.util.mark_negation
    # Thus we change n't or n' to not, for further cleaning
    text = re.sub("[nN]'[tT]", " not", text)
    text = re.sub("[nN]'", " not", text)

    # Change any punctuation repeat more than once to period.
    text = re.sub(r'[^A-Za-z0-9_\ ]{2,}', " . ", text) # reduce ! ?

    # mark_negation function does not recognize comma as seperator but so does period
    text = re.sub(',', " . ", text)

    # Remove irrelavent characters
    text = re.sub(r'[^A-Za-z_\ .!?;:]', " ", text)

    # Tokenizing starts
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if
              word.lower() not in stopwords]
    # for sentence in nltk.sent_tokenize(text):
    #     for word in nltk.word_tokenize(sentence):
    #     # for word in sentence.split():
    #         if word.isalpha(): # If any alphabet exists in word. e.g. don't
    #             tokens.append(word.lower()) if word.lower() not in stopwords else None
    #         elif word in ';:.?!': # Leave out the punctuation except for :;.?!
    #             tokens.append(word)

    ### Negation ###
    # Deal With Negation: Add _NEG from the negative words to the nearest punctuation
    tokens = nltk.sentiment.util.mark_negation(tokens)

    # Only Keep those with word which at least contains two letters, '?' and '!'
    tokens = [token for token in tokens if re.search('[a-zA-Z]+.*[a-zA-Z]+', token) or re.search('[!?]', token)]

    ### FEATURE: good / bad words ###
    good_num = sum(word in good_word for word in tokens)
    bad_num = sum(word in bad_word for word in tokens)

    ##### STEMMING ######
    # Two seperators for noun and verb respectively
    tokens = [porter_stemmer.stem(word) for word in tokens]
    stems = [wordnet_lemmatizer.lemmatize(word, pos='v') for word in tokens]  # Verb

    return (attitude, good_num, bad_num, stems)


def token_wrapper(text):  # Wrap up the tokenizer
    attitude, good_num, bad_num, stems = tokenization_and_stemming_all(text)
    return stems




# Read in Data
ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/data/review_10k.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

df = pd.DataFrame(out)






##### 2. FEATURE SELECTION #####


### 2.0 CountVectorizer ###
vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=token_wrapper, max_features = 4000)
dat_x, dat_y = vect.fit_transform(df['text']), df['stars']
name = vect.get_feature_names() # Get the word title


### 2.1 DOCUMENT FREQUENCY (DF) ###

# p.s. We have already used DF in function CountVectorizer...
ans_DF =  np.apply_along_axis(lambda x: sum(1 for item in x if item > 0) , 0, dat_x.toarray()) / dat_x.shape[0]
combine_DF = [(name[i], ans_DF[i]) for i in range(len(ans_DF))]
combine_DF.sort(key=lambda x: x[1], reverse=True)


### 2.2 CHI-SQUARED ###

# The result is a piece of shit, and it's quite slow.
# We won't apply this method in our final analysis
from scipy.stats import chi2_contingency
def CHI(X,y):
    def one_class_CHI(X, y, num):
        y_tmp = (y == num).astype(float)
        X_tmp = (X > 0).astype(float).toarray()

        def CHI_with_y(X):
            con_table = pd.crosstab(X, y_tmp)
            _, p, _, _ = chi2_contingency(con_table)
            return p

        return np.apply_along_axis(CHI_with_y, 0, X_tmp)

    def CHI_classes(num):
        return one_class_CHI(X, y, num)

    l = list(map(CHI_classes, [1, 2, 3, 4, 5]))

    return np.concatenate(l,axis = 0).reshape(5,-1).mean(axis = 0)

ans_CHI = CHI(dat_x,dat_y)
combine_CHI = [(name[i], ans_CHI[i]) for i in range(len(ans_CHI))]
combine_CHI.sort(key=lambda x: x[1], reverse=True)


### 2.3 INFORMATION GAIN (IG) ###

# Codes are from Stackoverflow
# https://stackoverflow.com/questions/25462407/fast-information-gain-computation
# Time: very fast
def information_gain(X, y):
    def _calIg():
        entropy_x_set = 0
        entropy_x_not_set = 0
        for c in classCnt:
            probs = classCnt[c] / float(featureTot)
            entropy_x_set = entropy_x_set - probs * np.log(probs)
            probs = (classTotCnt[c] - classCnt[c]) / float(tot - featureTot)
            entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        for c in classTotCnt:
            if c not in classCnt:
                probs = classTotCnt[c] / float(tot - featureTot)
                entropy_x_not_set = entropy_x_not_set - probs * np.log(probs)
        return entropy_before - ((featureTot / float(tot)) * entropy_x_set
                                 + ((tot - featureTot) / float(tot)) * entropy_x_not_set)

    tot = X.shape[0]
    classTotCnt = {}
    entropy_before = 0
    for i in y:
        if i not in classTotCnt:
            classTotCnt[i] = 1
        else:
            classTotCnt[i] = classTotCnt[i] + 1
    for c in classTotCnt:
        probs = classTotCnt[c] / float(tot)
        entropy_before = entropy_before - probs * np.log(probs)

    nz = X.T.nonzero()
    pre = 0
    classCnt = {}
    featureTot = 0
    information_gain = []
    for i in range(0, len(nz[0])):
        if (i != 0 and nz[0][i] != pre):
            for notappear in range(pre + 1, nz[0][i]):
                information_gain.append(0)
            ig = _calIg()
            information_gain.append(ig)
            pre = nz[0][i]
            classCnt = {}
            featureTot = 0
        featureTot = featureTot + 1
        yclass = y[nz[1][i]]
        if yclass not in classCnt:
            classCnt[yclass] = 1
        else:
            classCnt[yclass] = classCnt[yclass] + 1
    ig = _calIg()
    information_gain.append(ig)

    return np.asarray(information_gain)

ans_IG = information_gain(dat_x, dat_y)
combine_IG = [(name[i], ans_IG[i]) for i in range(len(ans_IG))]
combine_IG.sort(key=lambda x: x[1], reverse=True)


### 2.4 POINTWISE MUTUAL INFORMATION (PMI) ###
# Use package pyitlib
def PMI(X,y):
    def one_class_PMI(X, y, num):
        y_tmp = (y == num).astype(float)
        X_tmp = (X > 0).astype(float).toarray()

        def MI_with_y(X):
            return drv.information_mutual(X, y_tmp)

        return np.apply_along_axis(MI_with_y, 0, X_tmp)

    def PMI_classes(num):
        return one_class_PMI(X, y, num)

    l = list(map(PMI_classes, [1, 2, 3, 4, 5]))

    return np.concatenate(l,axis = 0).reshape(5,-1).mean(axis = 0)

ans_PMI = PMI(dat_x,dat_y)
combine_PMI = [(name[i], ans_PMI[i]) for i in range(len(ans_PMI))]
combine_PMI.sort(key=lambda x: x[1], reverse=True)


# After comparing these four methods, IG is the fastest and the most accurate
# Select top 1000 features
feature_IG = set(x[0] for x in combine_IG[:1000])
feature_idx = [i for i in range(len(name)) if name[i] in feature_IG]
feature_name = [name[i] for i in range(len(name)) if name[i] in feature_IG]





##### 3. MODELING #####

# keywords = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/data/IG_result.csv')['word'])
# start = time.time()
# text = out[0]['text']
# att,good,bad,tokens = tokenization_and_stemming_all(text)
# dict_tmp = dict((x,[0]) for x in keywords)
# for item in tokens:
#     if item in keywords:
#         dict_tmp[item][0] += 1
# pd.DataFrame.from_dict(dict_tmp)
# end = time.time()
# print(end - start)



### 3.0 PREPROCESSING ###
dat = pd.DataFrame(dat_x[:,feature_idx].toarray())
dat.columns = feature_name

# Manually selected features
attitude, good, bad, dat_gnb = [], [], [], pd.DataFrame()
for item in df['text'].apply(tokenization_and_stemming_all):
    attitude.append(item[0])
    good.append(item[1])
    bad.append(item[2])
dat['good_count'] = good
dat['bad_count'] = bad
dat['attitude_count'] = attitude


## 3.0.1 TF-IDF ##
tfidf = sklearn.feature_extraction.text.TfidfTransformer()
dat_tfidf = tfidf.fit_transform(dat)
dat = pd.DataFrame(dat_tfidf.todense())


## 3.0.2 Boolean version ##
dat = (dat > 0).astype(float)




### 3.1 Naive Bayes ###
# Learning Materials: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies
# Train-Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat, dat_y.values,test_size=0.2, random_state = 960512)


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)


# Presentation-1
# Precision, recall and F1.
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(y_test, pre)
# recall = np.diag(cm) / np.sum(cm, axis = 1)
# precision = np.diag(cm) / np.sum(cm, axis = 0)
# F1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(5)]
# print('precision')
# print(np.mean(precision))
# print('recall')
# print(np.mean(recall))
# print('F1')
# print(np.mean(F1))

# Below code is for regression -> classification
# for i in range(len(pre)):
#     if pre[i] < 1 :
#         pre[i] = 1
#     elif pre[i] > 5:
#         pre[i] = 5
#     else:
#         pre[i] = np.round(pre[i])


### 3.2 Linear Model ###


clf = sklearn.linear_model.LinearRegression()
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)

### 3.3 Logistic Model ###

clf = sklearn.linear_model.LogisticRegression()
clf.fit(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)



### 3.4 SVM ###

from sklearn import svm
clf = svm.SVC(kernel='linear')
clf.fit(X_train, y_train)

pre_SVM = clf.predict(X_test)
res = [pre_SVM[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)




