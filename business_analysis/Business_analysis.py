# Import essencial packages
import json
import numpy as np
import pandas as pd
import collections
import time
import matplotlib
import matplotlib.pyplot as plt


# Try to find the merchant with most reviews
# Time: 0.009 for 1000 data

###### DATA PREPROCESSING ######
ccc = collections.defaultdict(int)
with open('/Users/moran/Google_Drive/Course/628/Proj2/review_train.json', 'r') as fh:
    for line in fh:
        d = json.loads(line)
        biz_id = d['business_id']
        ccc[biz_id] += 1

with open('ccc.txt', 'w') as file:
    file.write(json.dumps(ccc))

with open('ccc.txt', 'r') as file:
    for line in file:
        ccc = json.loads(line)

import heapq
ans = heapq.nlargest(10, ccc.items(), key = lambda x: x[1]) # Top 10
merchant_set = set(int(item[0]) for item in ans)
ans = dict((int(x),y) for x, y in ans)



out = []
with open('/Users/moran/Google_Drive/Course/628/Proj2/business_train.json', 'r') as fh:
    for line in fh:
        d = json.loads(line)
        biz_id = d['business_id']
        if biz_id in merchant_set:
            out.append(d)
merchant_top = pd.DataFrame(out)
merchant_top['review'] = [ans[x] for x in merchant_top['business_id']]
merchant_top = merchant_top.sort_values(by = ['review'], ascending=False)
merchant_top['stars'] = [4, 3.5, 4, 4, 4.5, 3.5, 4, 3.5, 2.5, 4.5]
merchant_top['strip'] = [(item in set([1,2,3,4,5,7,8])) + 0 for item in range(1,11)]
merchant_top_plot = merchant_top[['name','city','state', 'review', 'latitude', 'longitude', 'business_id', 'stars', 'strip']]
merchant_top_plot.to_csv('merchant_top.csv', index = False)

# One of them is closed: Gordon Ramsay BurGR 10459
merchant_set = set([138017, 94238, 60865, 154324, 113301])


###### Difference from BUSINESS ######
selected = [item in merchant_set for item in merchant_top['business_id']]
merchant_selected = merchant_top.iloc[selected,:]
# 1. Deal with attributes
df_tmp = pd.concat([pd.DataFrame([merchant_selected['attributes'].iloc[i]]) if not pd.isnull(merchant_selected['attributes'].iloc[i]) else pd.Series(np.nan) for i in range(5)], sort = False)

# 1.1 Deal with Ambience
import ast
df_tmp2 = pd.concat([pd.DataFrame([ast.literal_eval(df_tmp['Ambience'].iloc[i])]) if not pd.isnull(df_tmp['Ambience'].iloc[i]) else pd.Series(np.nan) for i in range(5)], sort = False)
df_tmp2 = df_tmp2.rename(mapper = lambda x: 'Ambience_' + x, axis='columns')

# 1.2 Deal with BusinessParking
df_tmp3 = pd.concat([pd.DataFrame([ast.literal_eval(df_tmp['BusinessParking'].iloc[i])]) if not pd.isnull(df_tmp['BusinessParking'].iloc[i]) else pd.Series(np.nan) for i in range(5)], sort = False)
df_tmp3 = df_tmp3.rename(mapper = lambda x: 'BusinessParking_' + x, axis='columns')

# 1.3 Deal with GoodForMeal
df_tmp4 = pd.concat([pd.DataFrame([ast.literal_eval(df_tmp['GoodForMeal'].iloc[i])]) if not pd.isnull(df_tmp['GoodForMeal'].iloc[i])  else pd.Series(np.nan) for i in range(5)], sort = False)
df_tmp4 = df_tmp4.rename(mapper = lambda x: 'GoodForMeal_' + x if x == str(x) else x, axis='columns')

# 1.4 Put them together

df_tmp = df_tmp.drop(['Ambience','BusinessParking','GoodForMeal'], axis=1)
df_att = pd.concat([df_tmp, df_tmp2, df_tmp3, df_tmp4], axis=1)

# 2. Deal with hours
df_hours = pd.concat([pd.DataFrame([merchant_selected['hours'].iloc[i]]) if not pd.isnull(merchant_selected['hours'].iloc[i]) else pd.Series(np.nan) for i in range(5)], sort = False)

# 3. Put those together

merchant_selected = merchant_selected.drop(['attributes','hours'], axis=1)
df_biz = pd.concat([merchant_selected.reset_index(drop=True),df_hours.reset_index(drop=True), df_att.reset_index(drop=True)], axis=1)
nunique = df_biz.apply(pd.Series.nunique)
cols_to_drop = nunique[nunique == 1].index
df_biz = df_biz.drop(cols_to_drop, axis=1)
df_biz = df_biz.drop(['business_id','latitude', 'longitude','review', 0],axis = 1)

df_biz.to_csv('df_biz.csv', index=False)


##### Difference from REVIEW ######

ccc2 = collections.defaultdict(list)
with open('/Users/moran/Google_Drive/Course/628/Proj2/review_train.json', 'r') as fh:
    for line in fh:
        d = json.loads(line)
        biz_id = d['business_id']
        if biz_id in merchant_set:
            ccc2[biz_id].append(d)

with open('ccc2.txt', 'w') as file:
    file.write(json.dumps(ccc2))

with open('ccc2.txt', 'r') as file:
    for line in file:
        ccc2 = json.loads(line)

df_wic = pd.DataFrame(ccc2['138017'])
df_hash = pd.DataFrame(ccc2['94238'])
df_earl = pd.DataFrame(ccc2['60865'])
df_bac = pd.DataFrame(ccc2['154324'])
df_cos = pd.DataFrame(ccc2['113301'])

# Top 10 words in each category
# Automate
out = pd.DataFrame()
for i in range(1,6):
    df = df_wic.loc[df_wic['stars'] == i]
    vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=token_wrapper, max_features=1000)
    dat_x, dat_y = vect.fit_transform(df['text']), df['stars']
    name = vect.get_feature_names()  # Get the word title
    ans_DF = np.apply_along_axis(lambda x: sum(1 for item in x if item > 0), 0, dat_x.toarray()) / dat_x.shape[0]
    combine_DF = [(name[i], ans_DF[i]) for i in range(len(ans_DF))]
    combine_DF.sort(key=lambda x: x[1], reverse=True)
    out['star' + str(i)] = [item[0] for item in combine_DF]
# Automate does not give much information
# Manually selected by Xiaohan, only Top 5
c = collections.Counter(np.array(out).reshape(1,-1).tolist()[0])
# We only keep those only appear once
c_1 = dict((x,y) for x,y in c.items() if y == 1)
for i in range(5):
    s = set(out.iloc[:,i])
    s = set(x for x in s if x in c_1)
    with open('set' + str(i+1) + '.txt', "w") as f:
        for item in s:
            f.write(str(item) + "\n")

# Top 12 Significant variables selected by IG
df = df_wic
vect = sklearn.feature_extraction.text.CountVectorizer(tokenizer=token_wrapper, max_features=1000)
dat_x, dat_y = vect.fit_transform(df['text']), df['stars']
name = vect.get_feature_names() # Get the word title
ans_IG = information_gain(dat_x, dat_y)
# end = time.time()
# print(end - start)
combine_IG = [(name[i], ans_IG[i]) for i in range(len(ans_IG))]
combine_IG.sort(key=lambda x: x[1], reverse=True)

target_word = set(item[0] for item in combine_IG[:12])
name = vect.get_feature_names()
d = dict((x,[0,0,0,0,0]) for x in target_word)
d['sum'] = [0, 0, 0, 0, 0]
col_idx = [i for i in range(1000) if name[i] in target_word]
dat_wic_target = dat_x[:,col_idx]
for i in range(df_wic.shape[0]):
    cls = int(dat_y[i]) - 1
    d['sum'][cls] += 1
    for j in range(12):
        d[name[col_idx[j]]][cls] += (dat_wic_target[i,j] > 0) + 0

pd.DataFrame(d).to_csv('IG_word.csv', index=False)










