#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


def generate_dic(dic_input):
    
    dict_file = open(dic_input)
    lines = dict_file.readlines()
    dict_file.close()
    dic = {}
    for line in lines:
        key,value = line.split(' ')
        dic[key] = value[:-1]
    return dic


# In[ ]:


def generate_labels(data_input):
    labels = []
    f = open(data_input,'r')
    lines = f.readlines()
#     print(lines)
    for i,k in enumerate(lines):
        s = k.split('\t')           #################33
#         print(s,i,k)
        labels.append(int(s[0]))   ##### left one
    return labels


# In[ ]:


def generate_index(data_input):
    index = []
    f = open(data_input,'r')
    lines = f.readlines()
    for i,k in enumerate(lines):
        s = k.split('\t')
        index.append([int(right.split(':')[0]) for right in s[1:]])
        index[i].append(len(dic))              #################################################
        
    return index


# In[ ]:


def sparse_dot(index,w):
    z = 0
    for i in index:
        z += w[i]*1
    return z


# In[ ]:


def get_weight(index,labels,eta,epoch):
    w = np.zeros(len(dic)+1)         ####################################33
    for i in range(int(epoch)):
        for index_individual, z in zip(index,labels):
            
            feature_vector = np.zeros(len(dic)+1)           ###############################
            feature_vector[index_individual] = 1
            product = sparse_dot(index_individual,w)
            w += eta * feature_vector * (z - (np.exp(product)/(1 + np.exp(product))))
    return w


# In[ ]:


def output(label_output_file,index,labels,w):
    f = open(label_output_file,'w')
    wrong = 0
    for index_individual, z in zip(index, labels):
        product = sparse_dot(index_individual,w)
        sigmoid = np.exp(product)/(1+np.exp(product))
        if sigmoid >= 0.5:
            z_ = 1
        else:
            z_ = 0
        if z_ != z:
            wrong += 1
        f.write(f'{z_}\n')
    f.close()
    return (wrong)/len(index)


# In[ ]:


import sys


# In[ ]:


if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dic_input = sys.argv[4]
    train_output = sys.argv[5]
    test_output = sys.argv[6]
    metrics_output = sys.argv[7]
    epoch = sys.argv[8]
    
    dic = generate_dic(dic_input)  ##############
    train_labels = generate_labels(train_input)
    train_index = generate_index(train_input)
    test_labels = generate_labels(test_input)
    test_index = generate_index(test_input)
    eta = 0.1
    weight = get_weight(train_index,train_labels,eta,epoch)
    er_train = output(train_output,train_index,train_labels,weight)
    er_test = output(test_output,test_index,test_labels,weight)
    
    f = open(metrics_output,'w')
    f.write(f'error(train): {er_train}\n')
    f.write(f'error(test): {er_test}\n')
    f.close()
    

