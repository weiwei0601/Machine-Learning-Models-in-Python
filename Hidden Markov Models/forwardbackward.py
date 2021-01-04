#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def predict_single_sample(Pi_,A_,B_,word_single,tag_single,index_tag):
    v  = np.zeros((A_.shape[0], len(word_single)))
    w  = np.zeros((A_.shape[0], len(word_single)))
    p  = np.zeros((A_.shape[0], len(word_single)))
    
    v[:, 0] = np.log(Pi_ * B_[:, word_index_set[word_single[0]]])
    w[:, 0] = np.log(Pi_ * B_[:, word_index_set[word_single[0]]])
    p[:, 0] = np.array([i for i in range(len(Pi_))])
    
    
    for t in range(1, len(word_single)):
        v[:,t] = np.log(B_[:,word_index_set[word_single[t]]] * (((A_).T) @ np.exp(v[:,t-1])))
    for t in range(1, len(word_single)):
        for j in range(A_.shape[0]):
            w_compare = []
            for k in range(A_.shape[0]):
                w_compare.append(np.log(A_[k, j] * B_[j, word_index_set[word_single[t]]]) + w[k, t-1])
            w[j, t] =  np.max(w_compare)
            p[j, t] =  np.argmax(w_compare)
    pLast = np.argmax(w[:, -1])
    pred_li = []
    for t in range(len(word_single)-1, -1, -1):
        pred_li.append(pLast)
        pLast = int(p[pLast, t])
    pred_li.reverse()
#     print('alpha',v)
    m = max(v[:,-1])
    predsingle, stasingle = [word_single[i]+'_'+index_tag[pred_li[i]] for i in range(len(pred_li))], [index_tag[pred_li[i]] for i in range(len(pred_li))]
    sum_v_T = m + np.log(sum(np.exp(v[:,-1] - m )))
#     print('return',predsingle, stasingle,sum_v_T)
    return predsingle, stasingle,sum_v_T


# In[3]:


def generate_prediction(Pi_,A_,B_,word_set,tag_set,index_tag):
    pred_set, sta_set = [], [] #initialize
    total_likelihood = 0
    for i in range(len(tag_set)):   #for each sample(line)
        word_single = word_set[i]
        tag_single = tag_set[i]
        Predsingle, Stasingle, Sum_alpha_T= predict_single_sample(Pi_,A_,B_,word_single,tag_single,index_tag)
        pred_set.append(Predsingle)
        sta_set.append(Stasingle)
        total_likelihood += Sum_alpha_T
    aver_likelihood = total_likelihood/len(tag_set)
    return pred_set, sta_set, aver_likelihood


# In[5]:


import sys
from learnhmm import generate_word_label_set, generate_index


# In[ ]:


if __name__ == '__main__':
    test_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]
    
    PPPi = np.genfromtxt(hmmprior)
    AAA = np.genfromtxt(hmmtrans)
    BBB = np.genfromtxt(hmmemit)
    
    word_set, tag_set = generate_word_label_set(test_input)
    word_index_set =  generate_index(index_to_word)
    tag_index_set =  generate_index(index_to_tag)
    
    index_tag = dict((i,j)for j,i in tag_index_set.items())
    
    Predict_set, Sta_set, Aver_likelihood = generate_prediction(PPPi,AAA,BBB,word_set,tag_set,index_tag)
    
    f1 = open(predicted_file,'w')
    for sample in Predict_set:
        for item in range(len(sample)):
            if item != len(sample) - 1:
                f1.write(sample[item]+' ')
            else:
                f1.write(sample[item])
        f1.write('\n')
        
    f1.close()
    
    total = 0
    correct = 0
    for sample in range(len(tag_set)):
        for item in range(len(tag_set[sample])):
            total += 1
            if tag_set[sample][item] == Sta_set[sample][item]:
                correct += 1
                
    acc = correct/total     
    
    f2 = open(metric_file,'w')
    f2.write(f'Average Log-Likelihood: {Aver_likelihood}')
    f2.write('\n')
    f2.write(f'Accuracy: {acc}')
    f2.write('\n')
    f2.close()
    
    

