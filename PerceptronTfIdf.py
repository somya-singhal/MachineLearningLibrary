
# coding: utf-8

# # Term frequency-inverse document frequency (tf-idf) representation 

# In[451]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import time
from scipy.sparse import hstack
import math


# In[452]:

traindata=pd.read_csv('reviews_tr.csv')
testdata=pd.read_csv('reviews_te.csv')


# In[453]:

train=traindata.head(500000)
target_train=train['label']
del train['label']
test=testdata
target_test=test['label']
del test['label']


# In[454]:

traincv=CountVectorizer()
traincv_matrix=traincv.fit_transform(train['text'])
testcv_matrix=traincv.transform(test['text'])


# In[455]:

traincv_matrix_col = traincv_matrix.tocsc()


# In[456]:

idf_count = list()
for col in range(traincv_matrix.shape[1]):
        D_count = len(traincv_matrix_col[:,col].nonzero())
        D_count_new = math.log10(500000 / D_count)
        idf_count.append(D_count_new)


# In[457]:

traincv_matrix = traincv_matrix.multiply(idf_count)


# In[458]:

target_train = target_train.replace(0, -1)
target_test = target_test.replace(0, -1)


# In[459]:

traincv_matrix = traincv_matrix.tocsr()


# In[460]:

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


# In[461]:

start = time.time()
tW = weights(traincv_matrix,target_train)
print(time.time()-start)


# In[462]:

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


# In[463]:

print("Train_accuracy is", correct/500000)
print("Train_error is", wrong/500000)


# In[464]:

testcv_matrix = testcv_matrix.multiply(idf_count)
testcv_matrix = testcv_matrix.tocsr()


# In[465]:

correct1=0
wrong1=0
for i in range(testcv_matrix.shape[0]):
    x = testcv_matrix[i].toarray()
    x = np.append(x,[[1]])
    pred_x = tW.dot(x)
    if pred_x>0:
        pred_x = 1
    
    else:
        pred_x = -1
    
    if pred_x == target_test[i]:
        correct1 += 1
    else:
        wrong1 += 1
        


# In[466]:

print("Test_accuracy is", correct1/(testcv_matrix.shape[0]))
print("Test_error is", wrong1/(testcv_matrix.shape[0]))


# In[479]:

ind_max = np.argpartition(tW[0].tolist()[0], -10)[-10:]
ind_min = np.argpartition(tW[0].tolist()[0], 10)[:10]


# In[480]:

print("Highest weight words: ")
for i in ind_max:
    print(traincv.get_feature_names()[i]) 


# In[481]:

print("Lowest weight words: ")
for i in ind_min:
    print(traincv.get_feature_names()[i])


# In[ ]:



