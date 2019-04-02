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
import time

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
    return (attitude, good_num, bad_num)


###### TEST DATA ######
ourfile = '/Users/moran/Google_Drive/Course/628/Proj2/review_test.json'
out = []
with open(ourfile, 'r') as fh:
    for line in fh:
        d = json.loads(line)
        out.append(d)

out2 = out[:1000]
df_test = pd.DataFrame(out2)

def only_ratio(text):
    attitude, good_num, bad_num = tokenization_and_stemming_all(text)
    if good_num == 0:
        if bad_num == 0:
            return 3
        else:
            return 1.33
    elif bad_num == 0:
        return 4.66
    else:
        return (1 * bad_num + 5 * good_num) / (good_num + bad_num)

start = time. time()
with open('submit.csv', 'w') as f:
    f.write('ID,Expected\n')
    for i, j in enumerate(map(only_ratio, df_test['text'])):
        f.write("{},{}\n".format(i + 1, j))
end = time. time()
print(end - start)
# Time use: 2933s = 48min





