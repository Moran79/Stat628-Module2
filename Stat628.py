# Import Essential Packages
import json
import sklearn
import collections
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nltk
import re
import textblob
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
import nltk.sentiment
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from pyitlib import discrete_random_variable as drv
import time


###### STOPWORDS ######
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

# Stemming Initialization
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


######## GOOD WORD AND BAD WORD ########
# Read in Pos & Neg dictionary by Minqing Hu and Bing Liu
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
good_word = set(
    pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/positive-words.txt', header=None,
                names=['word'])['word'])
bad_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/negative-words.txt',
                           encoding="ISO-8859-1", header=None, names=['word'])['word'])
# Add NEG dictionary
good_tmp = set(x + '_NEG' for x in bad_word)
bad_tmp = set(x + '_NEG' for x in good_word)
good_word = good_word | good_tmp
bad_word = bad_word | bad_tmp


###### TOKENIZING ######
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



###### FEATURE SELECTION ######

# Read in Data
ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/data/review_1k.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

df = pd.DataFrame(out)


#Time: 6887 * 4 = 39.74 s
vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=token_wrapper)
start = time.time()
dat_x, dat_y = vect.fit_transform(df['text']), df['stars']
name = vect.get_feature_names() # Get the word title
end = time.time()
print(end - start)


### 1. DOCUMENT FREQUENCY (DF) ###

# Although the result is promising, We have already used DF in function CountVectorizer
# Thus, We decide not to use twice
ans_DF =  np.apply_along_axis(lambda x: sum(1 for item in x if item > 0) , 0, dat_x.toarray()) / dat_x.shape[0]
combine_DF = [(name[i], ans_DF[i]) for i in range(len(ans_DF))]
combine_DF.sort(key=lambda x: x[1], reverse=True)

dat_plot = pd.DataFrame(combine_DF)
dat_plot.columns = ['word', 'DF']
threshold_DF = np.mean(dat_plot['DF']) + 1.645 * np.var(dat_plot['DF']) ** 0.5
dat_plot = dat_plot.iloc[:sum(1 for x in dat_plot['DF'] if x > threshold_DF), :]
dat_plot.plot()
plt.show()
feature_DF = set(dat_plot['word'])
len(feature_DF) # 47 features


### 2. CHI-SQUARED ###

# The result is a piece of shit, and it's quite slow.
# We won't apply this method in our final analysis
from scipy.stats import chi2_contingency
# Time: One word takes about 0.0596s (1000 rows)

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
combine_CHI.sort(key=lambda x: x[1])

dat_plot = pd.DataFrame(combine_CHI)
dat_plot.columns = ['word', 'CHI']
threshold_CHI = 0.1
dat_plot = dat_plot.iloc[:sum(1 for x in dat_plot['CHI'] if x <= threshold_CHI), :]
dat_plot.plot()
plt.show()
feature_CHI = set(dat_plot['word'])
len(feature_CHI) # 20 features



### 3. INFORMATION GAIN ###

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

# Time: One word takes about 0.00046285s (1000 rows)
# start = time.time()
ans_IG = information_gain(dat_x, dat_y)
# end = time.time()
# print(end - start)
combine_IG = [(name[i], ans_IG[i]) for i in range(len(ans_IG))]
combine_IG.sort(key=lambda x: x[1], reverse=True)

dat_plot = pd.DataFrame(combine_IG)
dat_plot.columns = ['word', 'IG']
threshold_IG = np.mean(dat_plot['IG']) + 1.645 * np.var(dat_plot['IG']) ** 0.5
dat_plot = dat_plot.iloc[:sum(1 for x in dat_plot['IG'] if x > threshold_IG), :]
dat_plot.plot()
plt.show()
feature_IG = set(dat_plot['word'])
len(feature_IG) # 55 features

# Select top 1000 features
feature_IG = set(x[0] for x in combine_IG[:1000])



### 4. POINTWISE MUTUAL INFORMATION (PMI) ###
# Use package pyitlib
# Time: One word takes about 0.0235s (1000 rows)
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

dat_plot = pd.DataFrame(combine_PMI)
dat_plot.columns = ['word', 'PMI']
threshold_PMI = np.mean(dat_plot['PMI']) + 1.645 * np.var(dat_plot['PMI']) ** 0.5
dat_plot = dat_plot.iloc[:sum(1 for x in dat_plot['PMI'] if x > threshold_PMI), :]
dat_plot.plot()
plt.show()
feature_PMI = set(dat_plot['word'])
len(feature_PMI) # 57 features


# As we can see that IG already capture 92% information
# PMI also runs very slow, thus we decide to just IG
feature_idx = [i for i in range(len(name)) if name[i] in feature_IG]
feature_name = [name[i] for i in range(len(name)) if name[i] in feature_IG]



###### MODELING ######

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



### Preprocessing ###
dat = pd.DataFrame(dat_x[:,feature_idx].toarray())
dat.columns = feature_name
dat_backup = dat.copy()
# dat = dat_backup[:]


# TF-IDF
dat = dat_x.copy()
tfidf = sklearn.feature_extraction.text.TfidfTransformer()
dat_tfidf = tfidf.fit_transform(dat)
dat = pd.DataFrame(dat_tfidf.todense())

from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
hv_model = HashingVectorizer(tokenizer = token_wrapper)
tfidf_model = TfidfVectorizer(max_df=0.8, max_features=10000,
                                 min_df=0.2,
                                 use_idf=True, tokenizer = token_wrapper, ngram_range=(1,1))

tfidf_matrix = tfidf_model.fit_transform(df['text'])
dat_x = tfidf_matrix

hv_matrix = hv_model.fit_transform(df['text'])
dat_x = tfidf_matrix

# Boolean version
dat = (dat > 0).astype(float)






### Naive Bayes ###
# Learning Materials: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies
# Wait for further tuning


dat_x = dat.copy()
# Train-Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat_x, dat_y.values,test_size=0.2)



from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)
plt.hist(res) # Show histogram
plt.show()

# from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pre)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
F1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(5)]

print('precision')
print(np.mean(precision))
print('recall')
print(np.mean(recall))
print('F1')
print(np.mean(F1))



### Linear Model ###


### Add several features ###
dat_lm = X_boolean[:]


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat, dat_y.values,test_size=0.2)

clf = sklearn.linear_model.LinearRegression()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)
plt.hist(res) # Show histogram
plt.show()


for i in range(len(pre)):
    if pre[i] < 1 :
        pre[i] = 1
    elif pre[i] > 5:
        pre[i] = 5
    else:
        pre[i] = np.round(pre[i])


cm = confusion_matrix(y_test, pre)
recall = np.diag(cm) / np.sum(cm, axis = 1)
precision = np.diag(cm) / np.sum(cm, axis = 0)
F1 = [2 / (1 / precision[i] + 1 / recall[i]) for i in range(5)]

print('precision')
print(np.mean(precision))
print('recall')
print(np.mean(recall))
print('F1')
print(np.mean(F1))

cm2 = cm[:]

### Just good and bad ###


X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat_gnb, dat_y.values,test_size=0.2)

clf = sklearn.linear_model.LinearRegression()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)
plt.hist(res) # Show histogram
plt.show()


start = time.time()

end = time.time()
print(end-start)

### Three model together ###
attitude, good, bad, dat_gnb = [], [], [], pd.DataFrame()
for item in df['text'].apply(tokenization_and_stemming_all):
    attitude.append(item[0])
    good.append(item[1])
    bad.append(item[2])

dat_gnb['good_count'] = good
dat_gnb['bad_count'] = bad
dat_gnb['attitude_count'] = attitude

X_train, X_test, y_train, y_test, X2_train, X2_test = sklearn.model_selection.train_test_split(dat, dat_y.values, dat_gnb,test_size=0.2)

#Naive Bayes

dat_x['good_count'] = good
dat_x['bad_count'] = bad
dat_x['attitude_count'] = attitude
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat_x, dat_y.values,test_size=0.2)
clf = MultinomialNB()
clf.fit(X_train, y_train)
pre_NB = clf.predict(X_test)
res = [pre_NB[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)
cm_NB = confusion_matrix(y_test, pre_NB)
precision_NB = np.diag(cm_NB) / np.sum(cm_NB, axis = 0)
print(precision_NB)

#LM
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat_x, dat_y.values,test_size=0.2)
clf = sklearn.linear_model.LinearRegression()
clf.fit(X_train, y_train)
pre_LM = clf.predict(X_test)
res = [pre_LM[i] - y_test[i] for i in range(len(y_test))]
# print((sum(x ** 2 for x in res) / len(res)) ** 0.5)

pre_LMR = pre_LM.copy()
for i in range(len(pre_LMR)):
    if pre_LMR[i] < 1 :
        pre_LMR[i] = 1
    elif pre_LMR[i] > 5:
        pre_LMR[i] = 5
    else:
        pre_LMR[i] = np.round(pre_LMR[i])

res = [pre_LMR[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)

cm_LMR = confusion_matrix(y_test, pre_LMR)
precision_LMR = np.diag(cm_LMR) / np.sum(cm_LMR, axis = 0)
print(precision_LMR)

# Just good and bad

clf = sklearn.linear_model.LinearRegression()
clf.fit(X2_train, y_train)
pre_GNB = clf.predict(X2_test)
res = [pre_GNB[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)

pre_GNBR = pre_GNB.copy()
for i in range(len(pre_GNBR)):
    if pre_GNBR[i] < 1 :
        pre_GNBR[i] = 1
    elif pre_GNBR[i] > 5:
        pre_GNBR[i] = 5
    else:
        pre_GNBR[i] = np.round(pre_GNBR[i])


cm_GNBR = confusion_matrix(y_test, pre_GNBR)
precision_GNBR = np.diag(cm_GNBR) / np.sum(cm_GNBR, axis = 0)
print(precision_GNBR)
print(precision_NB,'\n',precision_LMR,'\n',precision_GNBR)

df = pd.DataFrame()
df['pre_NB'] = pre_NB
df['pre_LM'] = pre_LM
df['pre_LMR'] = pre_LMR
df['pre_GNB'] = pre_GNB
df['pre_GNBR'] = pre_GNBR
df['real'] = y_test
df.to_csv('three_method.csv', index = False)
### Logistic Model ###



### SVM ###

import time
from sklearn import svm
from sklearn.metrics import classification_report
# Perform classification with SVM, kernel=linear
dat_x['good_count'] = good
dat_x['bad_count'] = bad
dat_x['attitude_count'] = attitude

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(dat_x, dat_y.values,test_size=0.2)
classifier_linear = svm.SVC(kernel='linear')
t0 = time.time()
classifier_linear.fit(X_train, y_train)

pre_SVM = classifier_linear.predict(X_test)
res = [pre_SVM[i] - y_test[i] for i in range(len(y_test))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)


with open('submit.csv', 'w') as f:
    f.write('ID,Expected\n')

start = time.time()
with open('/Users/moran/Google_Drive/Course/628/Proj2/data/review_test.json', 'r') as fh:
    for idx, line in enumerate(fh):
        d = json.loads(line)
        attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(d['text'])
        text = set(text)
        new_x = np.append([0 + (item in text) for item in feature_name],
                          [good_count, bad_count, attitude_count]).reshape(1, -1)
        new_score = classifier_linear.predict(new_x)[0]
        with open('submit.csv', 'a') as f:
            f.write(str(idx+1) + ',' + str(new_score) + '\n')
end = time.time()
print(end - start)

# 1000 / 11s

print(sum([x**2 for x in res]))



# out.append(d)


#### SUBMISSION ####

with open('submit.csv', 'w') as f:
    f.write('ID,Expected\n')


vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=token_wrapper)

dat_x = vect.fit_transform(d['text'])
name = vect.get_feature_names() # Get the word title



### MaxEnt ###




###### Pipeline ######

### Foreign Reviews ###

# For foreign review, just guess as 3.7
# We also find some European language, there are some meaningful word in it.
# We want to keep those, thus we define a metric
# len(text.encode('utf-8')) / len(text)
# Because we know that a foreign character can be describe as a 3-bit character
# We run 10000 test sample and find 2 seems to be a reasonable seperater.

# ratio = []
# for i in range(10000):
#     text = df['text'][i]
#     ratio.append(len(text.encode('utf-8')) / len(text))
#
# [i for i in range(10000) if ratio[i] > 1.05]
#
# import heapq
# answer = heapq.nlargest(10,enumerate(ratio), key = lambda x: x[1])
# [df['text'][item[0]] for item in answer]


# Word2Vec --- It's more about machine-learning, let's just leave it here.
import gensim
from gensim.models import Word2Vec


###### TEST DATA ######
# Evaluate the performance of tokenizing with manual check
# ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/review_1k.json'
# out = []
# with open(ourfile, 'r') as fh:
#     for line in fh:
#         d = json.loads(line)
#         out.append(d)
#
# df_test = pd.DataFrame(out)
# # out1.to_csv('try.csv', index=False)
#
# res = []
# for i in range(10):
#     text = df_test['text'][i]
#     res.append(text)
#     res.extend(['\n'])
#     res.append(token_wrapper(text))
#     res.extend(['\n','\n','\n'])
#
# with open('token_test.txt', 'w') as f:
#     for item in res:
#         f.write("%s\n" % item)
# text = "I don't like this &%^$#)_.;'> place,,, but my father told me not to GIVE UP !!!! I felt very good and happy "
# text = "I love this FANTASTIC restaurant !!!"
# tokenization_and_stemming_all(text)
###### 测试集结束 ######


###### Lexicon-Based Sentiment Analysis ######

# 中文称作词典法
# Predicition RMSE: 1.0+
# Can not detect sarcasm or other form of emotion transitions.
# It is a real tough question, let's just first leave it here.

# sklearn.hashingvetorize 培根的方法


ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/data/review_1k.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

df = pd.DataFrame(out)

df_search = df.iloc[idx]
d = df_search.iloc[2,:]



word_plot = [(feature_name[i], new_x[0][i]) for i in range(len(feature_name))]
import heapq
heapq.nlargest(30, word_plot, key = lambda x: x[1])



#### cut the second half of the data
# Percent = 50%

ans = []
for j in ratio:
    tmp = 0
    for i in range(13):
        t = df_search.iloc[i,3]
        attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(t)
        text = set(text)
        new_x = np.append([0 + (item in text) for item in feature_name],
                          [good_count, bad_count, attitude_count]).reshape(1, -1)
        new_score = classifier_linear.predict(new_x)[0]

        t = df_search.iloc[i, 3]
        attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(t[int(len(t) * j):])
        text = set(text)
        new_x = np.append([0 + (item in text) for item in feature_name],
                          [good_count, bad_count, attitude_count]).reshape(1, -1)
        new_score2 = classifier_linear.predict(new_x)[0]
        if new_score != new_score2:
            tmp += 1
    ans.append((j, tmp))
print(ans)


for i in range(13):
    t = df_search.iloc[i, 3]
    attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(t)
    text = set(text)
    new_x = np.append([0 + (item in text) for item in feature_name],
                      [good_count, bad_count, attitude_count]).reshape(1, -1)
    new_score = classifier_linear.predict(new_x)[0]

    t = df_search.iloc[i, 3]
    attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(t[int(len(t) * 0.9):])
    text = set(text)
    new_x = np.append([0 + (item in text) for item in feature_name],
                      [good_count, bad_count, attitude_count]).reshape(1, -1)
    new_score2 = classifier_linear.predict(new_x)[0]
    print(new_score, new_score2)



lll = []
for i in range(1000):
    t = df.iloc[i,3]
    lll.append(len(t))
#
# 208
# 990
# 2968
# 594
# 456
# 462
# 165
# 1590
# 417
# 933
# 455
# 290
# 2256

plt.hist(np.array(lll))
plt.show()

test_text = d['text'][330:]


len(x for x in df['text'])

df['text'][31-1]
test_text = df['text'][24-1]

attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(test_text)
text = set(text)
new_x = np.append([0 + (item in text) for item in feature_name],
                  [good_count, bad_count, attitude_count]).reshape(1, -1)
new_score = classifier_linear.predict(new_x)[0]
print(attitude_count)
print(good_count)
print(bad_count)
print(new_score)


idx = [3, 9, 14, 17, 24, 26, 28, 31, 32, 33, 35, 41, 45, 48, 49, 54, 65, 66, 67, 68, 73, 75, 77, 83, 84, 85, 86, 87, 90, 91, 96, 102, 113, 114, 115, 117, 123, 124, 127, 129, 137, 139, 141, 152, 154, 155, 156, 157, 161, 162, 163, 166, 170, 172, 175, 176, 180, 184, 185, 189, 190, 193, 196, 203, 204, 207, 209, 212, 214, 220, 224, 229, 232, 233, 238, 239, 245, 246, 247, 253, 259, 261, 263, 266, 267, 268, 273, 276, 278, 279, 282, 300, 302, 304, 305, 313, 316, 319, 324, 328, 329, 330, 335, 337, 338, 342, 344, 348, 349, 350, 353, 357, 358, 359, 361, 363, 364, 366, 367, 368, 369, 373, 377, 379, 381, 382, 383, 385, 386, 390, 391, 392, 393, 397, 398, 400, 401, 402, 403, 410, 411, 414, 416, 420, 421, 427, 432, 436, 437, 441, 443, 444, 446, 449, 450, 454, 457, 458, 459, 460, 461, 463, 464, 467, 468, 469, 471, 479, 480, 481, 483, 484, 486, 488, 507, 511, 512, 514, 515, 520, 523, 526, 528, 529, 532, 533, 536, 543, 544, 545, 548, 549, 551, 552, 558, 569, 572, 573, 585, 588, 592, 597, 599, 600, 601, 603, 604, 615, 616, 617, 622, 624, 631, 638, 647, 650, 651, 661, 663, 666, 667, 668, 673, 674, 675, 676, 680, 681, 682, 685, 690, 692, 694, 700, 703, 704, 706, 711, 712, 713, 715, 716, 718, 720, 722, 725, 731, 732, 733, 735, 737, 739, 740, 742, 743, 745, 746, 748, 749, 750, 751, 754, 757, 759, 761, 764, 765, 769, 770, 774, 778, 779, 780, 783, 785, 792, 796, 800, 802, 803, 804, 807, 808, 815, 816, 817, 820, 821, 823, 825, 829, 835, 840, 848, 849, 853, 856, 860, 861, 863, 866, 878, 879, 881, 888, 890, 891, 892, 898, 901, 909, 912, 914, 921, 922, 923, 924, 925, 930, 932, 935, 937, 938, 945, 946, 950, 952, 956, 958, 959, 963, 966, 969, 970, 972, 973, 978, 980, 981, 982, 987, 989, 995]
idx2 = set([item - 1 for item in idx])


def find_unique(array):
    l = len(array)
    c = collections.Counter(array)
    for a,b in c.items():
        if b >= l-1:
            return (1,a)
    return (0,0)

start = time.time()
with open('/Users/moran/Google_Drive/Course/628/Proj2/data/review_1k.json', 'r') as fh:
    for idx, line in enumerate(fh):
        d = json.loads(line)
        ratio = [int(x / 10 * len(d['text'])) for x in range(10)]  # Search all possible segmentation
        def calculate_result(text):
            attitude_count, good_count, bad_count, text = tokenization_and_stemming_all(text)
            text = set(text)
            new_x = np.append([0 + (item in text) for item in feature_name],
                              [good_count, bad_count, attitude_count]).reshape(1, -1)
            new_score = classifier_linear.predict(new_x)[0]
            return new_score
        score1 = calculate_result(d['text'])
        score4 = calculate_result(d['text'][ratio[4]:])
        if score1 == score4:
            final_score = score1
        else:
            score_tmp = []
            for i in range(5,10):
                score_tmp.append(calculate_result(d['text'][ratio[i]:]))
                exist, num = find_unique(score_tmp)
                if exist:
                    final_score = num
                else:
                    final_score = (score1 + score4) / 2
        with open('submit.csv', 'a') as f:
            f.write(str(idx+1) + ',' + str(final_score) +'\n')

end = time.time()
print(end - start)

tmp = pd.read_csv('/Users/moran/PycharmProjects/leetcode/submit.csv', header = None)
y_pred = np.array(tmp[1])
res = [y_pred[i] - y[i] for i in range(len(y_pred))]
print((sum(x ** 2 for x in res) / len(res)) ** 0.5)




# 95% reviews less or equal to 360
import heapq
heapq.nlargest(50, ans)
plt.hist(ans)
plt.show()


### LSTM ###

wordsList = np.load('/Users/moran/Google_Drive/Course/628/Proj2/data/wordsList.npy')
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]
wordVectors = np.load('/Users/moran/Google_Drive/Course/628/Proj2/data/wordVectors.npy')
wordsList_set = set(wordsList)

start = time.time()
print(wordsList.index("i"))
end = time.time()
print(end - start)

ans = pd.DataFrame(ans)
plt.hist(ans)
plt.show()


text = d['text']

firstSentence = [wordsList.index(word.lower()) for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.lower() in wordsList_set]

import tensorflow as tf
with tf.Session() as sess:
    print(tf.nn.embedding_lookup(wordVectors,firstSentence).eval().shape)



def change_to_vector(text):
    tmp_list = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)][-360:]
    outfile = np.zeros((360), dtype='int32')
    for idx, word in enumerate(tmp_list):
        outfile[idx] = wordsList.index(word) if word in wordsList_set else 399999
    return outfile

#1000 data / 21s
start = time.time()
ids = np.zeros((1000, 360), dtype='int32')
ans = []
with open('/Users/moran/Google_Drive/Course/628/Proj2/data/review_1k.json', 'r') as fh:
    for idx,line in enumerate(fh):
        d = json.loads(line)
        text = d['text']
        ids[idx,:] = change_to_vector(text).copy()
end = time.time()
print(end - start)

### Useful functions ###

from random import randint

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(1,11499)
            labels.append([1,0])
        else:
            num = randint(13499,24999)
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        num = randint(11499,13499)
        if (num <= 12499):
            labels.append([1,0])
        else:
            labels.append([0,1])
        arr[i] = ids[num-1:num]
    return arr, labels


batchSize = 24
lstmUnits = 64
numClasses = 5
iterations = 100
maxSeqLength = 360
numDimensions = 50

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)





