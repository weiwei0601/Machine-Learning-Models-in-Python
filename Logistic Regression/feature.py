#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def generate_dic(dict_input):
    f = open(dict_input,'r')
    dic = {}
    for line in f:
        line_ = line.strip('\n')
        line_ = line_.split(' ')
        dic[line_[0]] = line_[1]   #define the key and value of the dictionary
    f.close()
    return dic


# In[3]:


def generate_model_1(dic,data_input,data_output):
    fi = open(data_input,'r')
    fo = open(data_output,'w')
    for line in fi:
        model = line[0] # first, input the label column
        dic_ = {}   #a new dictionary 
        line_ = line[2:].split(' ')
        for word in line_:  # the line_ is all values(words)        
            if word in dic:  #that word is also in the dict.txt
                dic_.setdefault(dic[word],'shiwei_xian') #replace it as 'the index of that word ' and whatever
            else:
                continue  #if the word is not appearing in the dictionary, do nothing
        # have the new dic_, time to write it in the format form
        for key in dic_.keys():
            model += '\t'
            model += key
            model += ':1'
        model += '\n'
        fo.write(model)
    fi.close()
    fo.close()
    return None    


# In[4]:


def generate_model_2(dic,data_input,data_output):
    fi = open(data_input,'r')
    fo = open(data_output,'w')
    for line in fi:
        model = line[0] 
        dic_ = {}   
        line_ = line[2:].split(' ')
        for word in line_:         
            if word in dic: 
                dic_.setdefault(dic[word],0)
                dic_[dic[word]] += 1  # this time let the value of the dic_ to be a counter.
            else:
                continue  
        
        for key in dic_.keys():
            if dic_[key] < 4:    # less than the threshold
                
                model += '\t'
                model += key
                model += ':1'
        model += '\n'
        fo.write(model)
    fi.close()
    fo.close()
    return None  


# In[ ]:


import sys


# In[ ]:


if __name__ == '__main__':
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dic_input = sys.argv[4]
    train_output = sys.argv[5]
    validation_output = sys.argv[6]
    test_output = sys.argv[7]
    flag = sys.argv[8]
    
    dic = generate_dic(dic_input)  ##############
    
    if flag == '1':
        generate_model_1(dic,train_input,train_output)
        generate_model_1(dic,validation_input,validation_output)
        generate_model_1(dic,test_input,test_output)
        
    else:
        generate_model_2(dic,train_input,train_output)
        generate_model_2(dic,validation_input,validation_output)
        generate_model_2(dic,test_input,test_output)

