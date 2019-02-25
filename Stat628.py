#Import Essential Packages
import json
import collections
import matplotlib
import numpy as np
import pandas as pd
import nltk
import re
from nltk.stem.snowball import SnowballStemmer
import textblob
# nltk.download('punkt')
# nltk.download('stopwords')
import nltk.sentiment



ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/review_q.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

text = d['text']

df = pd.DataFrame(out)
# out1.to_csv('try.csv', index=False)

#一些想做的事：
#对于全是大写的，或者多个叹号的，我们倾向于认为情感更极端一点。
#对于每个评价，我们可以用business找到对应的商家，对于不同的商业类型给与不同的答案/评价
#对于didn't like 我们可以变成Not_like

# Two Steps to extract information.
# 1. Delete all pronouns: Tokenizing
# 2. Delete all non-alphabets: Stemming

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




def tokenization_and_stemming(text):
    # Correct mis-spelling
    #Have not found any package which is promising
    attitude = 0  # Reviewer's attitude
    attitude += len(re.findall(r'[?!]+', text)) #If multiple ? or ! used, we assume that the reviewer has a strong attitude
    attitude += len(re.findall(r'[A-Z]{4,}', text))
    # All CAP words also indicate strong attitude.
    # I use four as boundary of the word length
    text = re.sub("[nN]'[tT]", " not", text)
    text = re.sub("[nN]'", " not", text)
    text = re.sub(r'[^A-Za-z0-9_\ ]{2,}', ".", text) # Change any punctuation repeat more than once to period.
    text = re.sub(',', ".", text) # mark_negation function below does not handle comma but period
    text = re.sub(r'[^A-Za-z0-9_\ .!?;:]', " ", text)
    #nltk.word_tokenize would split don't to do, n't.
    # But n't could not be recognized by nltk.sentiment.util.mark_negation
    #Thus we have to deal with patterns like don't

    tokens = []
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word.lower() not in stopwords]
    # for sentence in nltk.sent_tokenize(text):
    #     for word in nltk.word_tokenize(sentence):
    #     # for word in sentence.split():
    #         if word.isalpha(): # If any alphabet exists in word. e.g. don't
    #             tokens.append(word.lower()) if word.lower() not in stopwords else None
    #         elif word in ';:.?!': # Leave out the punctuation except for :;.?!
    #             tokens.append(word)
    # Deal With Negation: Add _NEG from negative words to the nearest punctuation
    tokens = nltk.sentiment.util.mark_negation(tokens)
    tokens = [token for token in tokens if re.search('[a-zA-Z]', token)] # Only Keep those with at least one letter
    return tokens


text = "I REALLY DON'T LIKE IT !!!!! I will never go again!!!!"
text = "What?!?!?! By the way, I will never go again???"
tokenization_and_stemming(text)


# Feature Selection on Pos & Neg words.

#Read in Pos & Neg dictionary by Minqing Hu and Bing Liu
#https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html#lexicon
good_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/positive-words.txt', header = None, names = ['word'])['word'])
bad_word = set(pd.read_csv('/Users/moran/Google_Drive/Course/628/Proj2/opinion-lexicon-English/negative-words.txt', encoding = "ISO-8859-1", header = None, names = ['word'])['word'])
#Stem
stemmer = SnowballStemmer("english")

def good_bad_stem(token):
    good_num = sum(word in good_word for word in token)
    bad_num = sum(word in bad_word for word in token)
    stems = [stemmer.stem(t) for t in token]
    return (good_num, bad_num, stems)

res = []
for i in range(10):
    text = df['text'][i]
    res.append(text)
    res.extend(['\n'])
    res.append(tokenization_and_stemming(text))
    res.extend(['\n','\n','\n'])

with open('your_file.txt', 'w') as f:
    for item in res:
        f.write("%s\n" % item)


#Naive Bayes
import sklearn

ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/review_1k.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

df = pd.DataFrame(out)
###Tokenizing




#Negation

#Stem
#问题是她把reasonable 变成了reason，先不管了




#Train-Test split
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(df['text'].values,
                 df['stars'].values,
                 test_size=0.2)

vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer = tokenization_and_stemming)
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



