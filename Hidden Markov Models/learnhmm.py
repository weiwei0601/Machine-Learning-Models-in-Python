#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def generate_word_label_set(data_path):
    f = open(data_path)
    word_set, label_set = [], [] #initialize
    for i in f:
        word_, label_ = [], []
        line = i.strip('\n').split(' ')
#         print('line',line)   ###########
        for item in line:
            word, label = item.split('_')
#             print('word',word)  #
#             print('label',label)  #
            word_.append(word)
            label_.append(label)
        word_set.append(word_)
        label_set.append(label_)
        
    return word_set, label_set


# In[3]:


def generate_index(index_path):
    #give Word and tag index from 0， 0,1,2,3...
    f = open(index_path)
    index_set = {}   # initialize dictionary
    amount = 0  # to count
    for i in f:
        index_set[i.strip('\n')] = amount
        amount +=1
        
    return index_set


# In[4]:


def generatePi(Tag_set,Tag_index_set,Pi_):
    for i in tag_set:
        Pi_[tag_index_set[i[0]]] += 1 
    Pi_ = (Pi_ + 1)/(sum(Pi_) + len(Pi_))
#     print('Pi',Pi_)
    return Pi_


# In[5]:


def generateA(Tag_set,Tag_index_set,A_):
    for i in Tag_set:
        for k in range(len(i) - 1):
            NI = k + 1
            TP = i[k]
            TN = i[NI]
            A_[Tag_index_set[TP],Tag_index_set[TN]] += 1
        
    A_ = (A_ + 1)/(np.sum(A_,axis=1).reshape((-1,1)) + A_.shape[1])
#     print('A',A_)
    return A_


# In[6]:


def generateB(Tag_set,Word_set,Tag_index_set,Word_index_set,B_):
    for LI in range(len(Tag_set)):
        for TI in range(len(Tag_set[LI])):
            word = Word_set[LI][TI]
            tag = Tag_set[LI][TI]
            B_[Tag_index_set[tag],Word_index_set[word]] += 1
            
    B_ = (B_ + 1)/(np.sum(B_,axis=1).reshape((-1,1)) + B_.shape[1])
#     print('B',B_)
    return B_


# In[7]:


import sys


# In[8]:


if __name__ == '__main__':
    train_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmmprior = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    
    word_set, tag_set = generate_word_label_set(train_input)
    word_index_set =  generate_index(index_to_word)
    tag_index_set =  generate_index(index_to_tag)
    
    #initialize A,B,Pi
    A =  np.zeros((len(tag_index_set),len(tag_index_set)))
    B =  np.zeros((len(tag_index_set),len(word_index_set)))
    Pi = np.zeros((len(tag_index_set),1))
    
    #get A,B,Pi
    PPi = generatePi(tag_set,tag_index_set,Pi)
    AA = generateA(tag_set,tag_index_set,A)
    BB = generateB(tag_set,word_set,tag_index_set,word_index_set,B)
    
    #write in file
    np.savetxt(hmmprior,PPi,delimiter=' ')
    np.savetxt(hmmtrans,AA,delimiter=' ')
    np.savetxt(hmmemit,BB,delimiter=' ')
    
    


# In[ ]:




