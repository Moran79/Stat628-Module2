#Import Essential Packages
import json
import sklearn
import collections
import matplotlib
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

#一些想做的事：
#对于每个评价，我们可以用business找到对应的商家，对于不同的商业类型给与不同的答案/评价
# Deal with mis-spelling


###### STOPWORDS ######
stopwords = set(nltk.corpus.stopwords.words('english')) #We choose some stopwords from library
keep1 = {"no","nor","not","don","don't", "shouldn't","ain","aren","aren't",'couldn',"couldn't",'didn','doesn',
        "doesn't","didn't","wouldn't",'wouldn',"won't","won","shan't","hasn't","hadn't",'hadn','hasn','haven',"haven't",
        "weren't",'weren',"wasn't",'wasn',"very","couldn't",'isn', 'mightn',"mightn't",'mustn',"mustn't", 'needn',"needn't",
        "isn't",'shan',"shan't", 'shouldn',"shouldn't",} #Negative words
keep2 = {"all","above","below","up","down"} #Something useful in stopwords
for word in keep1:
    stopwords.remove(word)
for word in keep2:
    stopwords.remove(word)
add1 = set("'ve") # Add some stopwords
for word in add1:
    stopwords.add(word)

# Stemming Initialization
porter_stemmer = PorterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()


######## GOOD WORD AND BAD WORD ########
# Read in Pos & Neg dictionary by Minqing Hu and Bing Liu
# https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
good_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/positive-words.txt', header = None, names = ['word'])['word'])
bad_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/negative-words.txt', encoding = "ISO-8859-1", header = None, names = ['word'])['word'])


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
    attitude += len(re.findall(r'[?!]+', text))



    ### TEXT CLEANING ###
    # nltk.word_tokenize would split [don't] to [do, n't]
    # But [n't] could not be recognized by nltk.sentiment.util.mark_negation
    # Thus we change n't or n' to not, for further cleaning
    text = re.sub("[nN]'[tT]", " not", text)
    text = re.sub("[nN]'", " not", text)

    # Change any punctuation repeat more than once to period.
    text = re.sub(r'[^A-Za-z0-9_\ ]{2,}', " . ", text)

    # mark_negation function does not recognize comma as seperator but so does period
    text = re.sub(',', " . ", text)

    # Remove irrelavent characters
    text = re.sub(r'[^A-Za-z_\ .!?;:]', " ", text)

    # Tokenizing starts
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.lower() not in stopwords]
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



def token_wrapper(text): # Wrap up the tokenizer
    attitude, good_num, bad_num, stems = tokenization_and_stemming_all(text)
    return stems



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
###### 测试集结束 ######


###### FEATURE SELECTION ######
# Read in Data
ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/review_1k.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

df = pd.DataFrame(out)
vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer = token_wrapper)
dat_x, dat_y = vect.fit_transform(df['text']), df['stars']

### 1. INFORMATION GAIN ###

# Codes are from Stackoverflow
# https://stackoverflow.com/questions/25462407/fast-information-gain-computation
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
                             +  ((tot - featureTot) / float(tot)) * entropy_x_not_set)

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
            for notappear in range(pre+1, nz[0][i]):
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

ans = information_gain(dat_x, dat_y)
name = vect.get_feature_names()
combine = [(name[i], ans[i]) for i in range(len(ans))]
combine.sort(key = lambda x: x[1], reverse = True)
combine[:20]





#Naive Bayes








#Train-Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df['text'].values,
                 df['stars'].values,
                 test_size=0.2)

vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer = token_wrapper)
tf_train = vect.fit_transform(X_train)
tf_test = vect.transform(X_test)

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
pre = clf.predict(X_test)
res = [pre[i] - y_test[i] for i in range(len(y_test))]
sum(x ** 2 for x in res)

res = [3.7 - y_test[i] for i in range(len(y_test))]
sum(x ** 2 for x in res)


#define vectorizer parameters
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_model = TfidfVectorizer(max_df= 0.99 , max_features=2000,
                                 min_df=0.1, stop_words='english',
                                 use_idf=True, tokenizer=tokenization_and_stemming, ngram_range=(1,1))

tfidf_matrix = tfidf_model.fit_transform(df['text'].values) #fit the vectorizer to synopses

#Train-Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(tfidf_matrix,
                 df['stars'].values,
                 test_size=0.2)


print ("In total, there are " + str(tfidf_matrix.shape[0]) + \
      " synoposes and " + str(tfidf_matrix.shape[1]) + " terms.")



