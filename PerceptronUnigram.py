
# coding: utf-8

# In[464]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import time
from scipy.sparse import hstack
import math


# In[465]:

traindata = pd.read_csv('reviews_tr.csv')
testdata = pd.read_csv('reviews_te.csv')


# In[466]:

train = traindata.head(500000)
target_train = train['label'].replace(0,-1)
del train['label']
test = testdata
target_test = test['label'].replace(0,-1)
del test['label']


# In[467]:

traincv = CountVectorizer()
traincv_matrix = traincv.fit_transform(train['text'])
testcv = CountVectorizer()
testcv_matrix = testcv.fit_transform(test['text'])


# In[468]:

def weights(data,labels):
    W=np.zeros((1,data.shape[1]+1))
    for j in range(2):
        idx=np.arange(data.shape[0])
        np.random.shuffle(idx)
        count=0
        total=W
        for i in idx:
            xtrain=data[i]
            xtrain=hstack([xtrain,[[1]]])
            value=labels[i]*xtrain.dot(W.T)
            if value <= 0:
                W = W + (labels[i] * xtrain)
                #print(W)
            if j == 1:
                total += W
            
            count += 1
            if count % 500000 == 0:
                print(" Pass {} completed {} data points".format(j, count))
        print("completed Pass")
    
    weight=total/(data.shape[0]+1)
    return weight
                


# In[470]:

start = time.time()
tW = weights(traincv_matrix, target_train)
print(time.time()-start)


# In[471]:

correct=0
wrong=0
for i in range(500000):
    x = traincv_matrix[i].toarray()
    x = np.append(x,[[1]])
    pred_x = tW.dot(x)
    if pred_x[0]>0:
        pred_x = 1
    
    else:
        pred_x = -1
    if pred_x == target_train[i]:
        correct += 1
    else:
        wrong+=1
        


# In[453]:

print("Unigram Train_accuracy is", correct/500000)
print("Unigram Train_error is", wrong/500000)


# In[454]:

train_feat = pd.DataFrame.from_dict(traincv.vocabulary_,orient='index').sort_values(by=[0])
test_feat = pd.DataFrame.from_dict(testcv.vocabulary_,orient='index').sort_values(by=[0])


# In[455]:

train_feat['features']=train_feat.index
test_feat['features']=test_feat.index
train_feat=train_feat.reset_index(drop=True)
test_feat=test_feat.reset_index(drop=True)


# In[456]:

tr = pd.DataFrame([[train_feat.shape[0],'Bias']],columns = [0, 'features'])
te = pd.DataFrame([[test_feat.shape[0],'Bias']],columns = [0, 'features'])


# In[457]:

train_feat = train_feat.append(tr,ignore_index = True)
test_feat = test_feat.append(te, ignore_index  = True)


# In[459]:

train_feat['weight']=tW.T


# In[460]:

testweight = pd.merge(test_feat, train_feat, on = 'features', how = 'left').fillna(0)


# In[462]:

testweight = testweight['weight']


# In[463]:

correct1=0
wrong1=0
for i in range(testcv_matrix.shape[0]):
    x = testcv_matrix[i].toarray()
    x = np.append(x,[[1]])
    pred_x = testweight.dot(x)
    if pred_x>0:
        pred_x = 1
    
    else:
        pred_x = -1
    
    if pred_x == target_test[i]:
        correct1 += 1
    else:
        wrong1 += 1
        


# In[441]:

print("Test_accuracy is", correct1/(testcv_matrix.shape[0]))
print("Test_error is", wrong1/(testcv_matrix.shape[0]))


# In[ ]:

ind_max = np.argpartition(tW[0].tolist()[0], -10)[-10:]
ind_min = np.argpartition(tW[0].tolist()[0], 10)[:10]


# In[ ]:

print("Highest weight words: ")
for i in ind_max:
    print(traincv.get_feature_names()[i]) 


# In[ ]:

print("Lowest weight words: ")
for i in ind_min:
    print(traincv.get_feature_names()[i])

