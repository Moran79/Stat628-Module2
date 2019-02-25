# 一些思路

我们是否需要先判断一句话是主观/客观呢？开始我想的是不需要，因为review都是主观的，但我后来想，难免会有一些客观的描述信息在里面呢？如果能把这些客观的信息去掉的话会不会更加准确？

有一个LDA可以根据主题分类？正好




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



# 具体做的步骤

## 随便猜猜

**只想用一个数字估计全部的评分**，为了最小化RMSE，需要用样本均值。开始试了用3分来估计, RMSE大约为1.62920，实在是不行。随后我发现因为样本是skewed，接下来我们面临第一个问题：**如何画出总体的histogram？** 这个我准备之后再放到server做，1000个数据跑了3.68秒，预计5364626行需要5.48小时。

暂时的解决方案是：我先选取了10组样本量为1000的数据，计算1~5星各自出现频率，计算出来10个样本均值，发现他们的median是3.7。用3.7估计总体之后发现RMSE变成了1.46367。我想这是我们在什么都不做的时候，能拿到的最好结果了。

这一万个数据中，1~5星出现的个数分别为：[1514, 812, 1062, 2250, 4362]

### 用到的文件

文件存在了Guess文件夹里面。

* calculate_hist.sh 是拿到10组1000
* fake.sh用来生成答案集
* submit.csv是答案集





