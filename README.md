# 一些思路

我们是否需要先判断一句话是主观/客观呢？开始我想的是不需要，因为review都是主观的，但我后来想，难免会有一些客观的描述信息在里面呢？如果能把这些客观的信息去掉的话会不会更加准确？

有一个LDA可以根据主题分类？正好

Find aspect/attribute/target of sentiment

虽然BiLSTM与attention几乎统治了NLP

https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#from-occurrences-to-frequencies

康贤盛2/26课

Yelp Fusion API

对于token的矩阵，设定一个范围，在这之内的单词才保留

Miss spelling 在token之后最后改一下

Foreign language not filtered/translated 【translation to English不同语言混杂不行】

Hidden Markov 可以取出一些Phrase

老师想查tea，但是查出来一堆中国火锅【奶茶】

Negative-binomial /Gamma 可以解释word-count distribution

老师在对于某一个特定单词出现次数上，用了boolean的方式，define了mean。

画了一个各个单词出现的五星频率图，画了一个各个茶种的五星频率图【Scaled Word Count 每一类除于各类星星】

如果关注一下service类的单词，会发现负面情绪过多。当我们对食物不满的时候，我们不会再关注事物本身，而是自己的情绪以及一些其他的不着边的东西

EDA (Experimental Data Analysis) Bonferroni correction ANOVA 去判断某个word是不是很关键

做完Naive Bayes / SVM / MaxEnt / Logistic / Linear Regression
TF-IDF 加权 / Boolean

## 其他组带来的Inspiration

Lemmatiza ? 这是啥 红麒麟组解释了将各种形式变成原型

LDA 李天奇组用了这个 Topics  Gibs sambling 来选择topics 画了一个蓝色和红色

他们还调查了用户是否return？怎么做到的[只是通过Back来判断的]

JST / Word2Vec

Mosaic plots 他们选了名词

把字符变成utf-8 去掉外国字符

emoji

adversative phrases  / clauses

Logistic what do you predict

trend model

NN: K-max









# Problems Encountered



1. nltk.word_tokenize(sentence) would change `don't` to `[do, n't]` . But `n't` could not be used by nltk.sentiment.util.mark_negation. Thus I decide to change all the `n't` to not.




# 第一阶段



## 数据基本信息

review_train.json: 5364626行

## 阅读材料

https://monkeylearn.com/sentiment-analysis/

搜索其他Sentiment Analysis

**任务：帮助老板提升revenue**

nlp: natural language processing

[Dan Jurafsky](https://www.youtube.com/watch?v=QIdB6M5WdkI&index=16&t=0s&list=WL) 说道：

* 负面评论出现很多negative words
* 根本都不提食物，全是感受
* stories in the past tense(我们希望把坏事情抛在脑后)
* Use we and us（We are gonna get through this bad thing together）



**sentiment analysis**

* (i) finding out what makes a review positive or negative 
* (ii) predicting a review’s rating based on its text and a small set of relevant attributes. 

We focus on **Polarity**: if the speaker express a *positive* or *negative* opinion

python nltk可以为词划分重不重要，但是词的中性/褒义/贬义

介绍视频里面的重要信息：

* 目前大概有75%的review被recommanded
* 那些被认为是以下三点的review不会被recommand，同时**也不计入分数计算**
  * Don't know them
  * Business Owner(Fake)
  * Ranter那些愤世嫉俗的人（我猜应该是总给差评的）





# 第二阶段

[本科生的predcition](https://www.kaggle.com/c/uw-madison-sp17-stat333/leaderboard)

[研究生的predcition](https://www.kaggle.com/c/uw-madison-sp18-stat628/leaderboard)

提交的时候两列分别为ID,Expected（中间不能有空格，也不能被引号括起来）

RMSE就是MSE的平方根



# Submission

## Guessing

**只想用一个数字估计全部的评分**，为了最小化RMSE，需要用样本均值。开始试了用3分来估计, RMSE大约为1.62920，实在是不行。随后我发现因为样本是skewed，接下来我们面临第一个问题：**如何画出总体的histogram？** 这个我准备之后再放到server做，1000个数据跑了3.68秒，预计5364626行需要5.48小时。

暂时的解决方案是：我先选取了10组样本量为1000的数据，计算1~5星各自出现频率，计算出来10个样本均值，发现他们的median是3.7。用3.7估计总体之后发现RMSE变成了1.46367。我想这是我们在什么都不做的时候，能拿到的最好结果了。

这一万个数据中，1~5星出现的个数分别为：[1514, 812, 1062, 2250, 4362]

### Result

* Use 3 to predict all: RMSE: 1.62920
* Use 3.7 to predict all: RMSE: 1.46367 [3.7 is sample average]

### 用到的文件

文件存在了Guess文件夹里面。

* calculate_hist.sh 是拿到10组1000
* fake.sh用来生成答案集
* submit.csv是答案集

## Use positive and negative words

建立一个好坏词词典，统计一个文本中好坏词的数量，然后用一个分段函数进行预测。具体预测方法如下表：

|    Prediction     | Good_num = 0 |          Good_num != 0          |
| :---------------: | :----------: | :-----------------------------: |
|  **Bad_num = 0**  |      3       |              1.33               |
| **Bad_num ! = 0** |     4.66     | Weighted average with ratio 1:5 |

### Run Time

2933.73 s $\approx$ 48.9 min

### Result

- RMSE: 1.10213

### File

Files are stored in Submission/Just_good_and_bad

* just_good_and_bad.py: Python code
* submit.csv: submission file





暂存网页：

https://www.yelp.com/login?return_url=%2Fdevelopers%2Fv3%2Fmanage_app

https://github.com/lwang535/STAT628_Module2_Group5/blob/master/code/produce_word_star_matrix.py

https://futrueboy.iteye.com/blog/944792

https://zhuanlan.zhihu.com/p/28053918

threshold for information gain

