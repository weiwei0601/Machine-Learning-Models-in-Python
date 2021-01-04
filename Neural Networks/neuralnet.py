#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def generate_data(input_path):
    f = open(input_path,'r')
    label = []
    data = []
    for line in f:
        l = line.split(',')[0]
        d = line.split(',')[1:]
        label += l
        data += [d]
    label_ = np.array(label).astype(np.int)   
    data_ = np.array(data).astype(np.int)
    os = np.ones((len(label_),1))
    os = np.array(os).astype(np.int)
#     print(os)
#     print(np.shape(os))
#     print(np.shape(data_))
    data_X = np.hstack((os,data_))
    
    data_Y = np.zeros((len(label_),10)).astype(np.int)
    for i in range(len(label_)):
        data_Y[i,label_[i]] = 1
    
    return label_, data_X, data_Y


# In[3]:


def weight_initialize(X,strategy,hidden_units):
    if strategy == 2:
        X_input_size = np.shape(X)[1]  # should be 128 + 1 = 129
        
        alpha = np.zeros((hidden_units, X_input_size)) 
        beta = np.zeros((10, hidden_units + 1)) # 10 , hidden_units 加上一个bias
        
    else:
        X_input_size = np.shape(X)[1] #129
        
        alpha_ = np.random.uniform(-0.1,0.1,(hidden_units, X_input_size-1)) # without bias 128
        bias_alpha = np.zeros((hidden_units, 1)) # the bias are all zeros
        alpha = np.hstack((bias_alpha , alpha_)) # combine them together
        #beta is similar
        beta_ = np.random.uniform(-0.1,0.1,(10, hidden_units))
        bias_beta = np.zeros((10,1))
        beta = np.hstack((bias_beta, beta_))
        
    return alpha, beta


# In[4]:


def NNF(X_single,Y_single,alpha,beta):  # forward
    
    a = alpha @ X_single
    
    z_ = 1/(1+np.exp(-a))
    bias_z = np.array([1])
    z = np.vstack((bias_z,  z_))
    
    b = beta@ z
    
    Y_hat = np.exp(b)/np.sum(np.exp(b))
    
    J = - np.sum(Y_single.T@ (np.log(Y_hat))) ######################### transpose Y_single

    
    return a, z_, z, b, Y_hat, J


# In[5]:


def NNB(X_single,Y_single,alpha,beta):   #backward
    obj = NNF(X_single,Y_single,alpha,beta)
    g_b  = obj[4] - Y_single # obj[4] = Y_hat
    g_beta = g_b @ obj[2].T  # obj[2] = z
    g_z = beta[:,1:].T @ g_b
    g_a = ((np.exp(-obj[0]))/(1+np.exp(-obj[0]))**2) * g_z # obj[0] = a
    g_alpha = g_a @ X_single.T
    
    return g_alpha, g_beta


# In[6]:


def SGD(X_single, Y_single, alpha, beta , lr):
    obj = NNF(X_single, Y_single, alpha, beta)
    g_alpha, g_beta = NNB(X_single, Y_single, alpha, beta)
    alpha -= lr * g_alpha
    beta -= lr * g_beta
    
    return alpha, beta


# In[7]:


def train(X, Y , alpha, beta, lr):
    for i in range(len(X)):
        X_single = X[i,:].reshape(-10,1)  # 行数是个负数就行
        Y_single = Y[i,:].reshape(-10,1)
        alpha, beta = SGD(X_single, Y_single, alpha, beta, lr)
    return alpha,beta


# In[8]:


def generate_label_loss(X, Y, alpha, beta):
    label = []
    loss = []
    for i in range(len(X)):
        X_single = X[i,:].reshape(-10,1)
        Y_single = Y[i,:].reshape(-10,1)
        obj = NNF(X_single, Y_single, alpha, beta)
        
        label_single = np.argmax(obj[4]) # obj[4] = Y_hat
        label.append(label_single)
        
        loss.append(obj[5]) # obj[5] = loss
        Loss = np.mean(loss)
    return label,Loss


# In[9]:


def error(label, label_):    # label- predict,  label_ - real
    errs = 0
    for i in range(len(label)):
        if label[i]!= label_[i]:
            errs += 1
    error_rate = errs/len(label)
    return error_rate


# In[10]:


import sys


# In[ ]:


if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    train_out = sys.argv[3]
    test_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])
    
    train_label_, X_train, Y_train = generate_data(train_input)
    test_label_, X_test, Y_test = generate_data(test_input)
    alpha, beta = weight_initialize(X_train, init_flag, hidden_units)  # initial alpha beta
    
    f = open(metrics_out,'w')
    
    for k in range(num_epoch):
        alpha, beta = train(X_train, Y_train, alpha, beta, learning_rate) # update of alpha and beta
        loss_train = generate_label_loss(X_train, Y_train, alpha, beta)[1]
        loss_test = generate_label_loss(X_test, Y_test, alpha, beta)[1]
        f.write(f'epoch={k+1} crossentropy(train): {loss_train}\n')
        f.write(f'epoch={k+1} crossentropy(test): {loss_test}\n')
        
    label_train = generate_label_loss(X_train, Y_train, alpha, beta)[0]
    label_test = generate_label_loss(X_test, Y_test, alpha, beta)[0]
    
    error_train = error(label_train, train_label_)
    f.write(f'error(train): {error_train}\n')
    
    error_test = error(label_test, test_label_)
    f.write(f'error(test): {error_test}\n')
    
    
    f1 = open(train_out,'w')
    for i in label_train:
        f1.write(f'{i}\n')
        
        
    f2 = open(test_out,'w')
    for i in label_test:
        f2.write(f'{i}\n')
        
    
    f.close()
    f1.close()
    f2.close()
    

