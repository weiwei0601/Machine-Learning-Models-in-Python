#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import sys
if __name__ == '__main__':
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    depth_max = int(sys.argv[3])      #把str 变 int，depth max
    train_output = sys.argv[4]
    test_output = sys.argv[5]
    metrics = sys.argv[6]


# In[ ]:


Train_old = np.loadtxt(fname = train_input,dtype='<U25')
Test_old  = np.loadtxt(fname = test_input,dtype='<U25')


# In[3]:


attribute_name = Train_old[0]


# In[4]:


Depth = 0                                   #初始深度为零

Train_old = Train_old[1:]
Test_old = Test_old[1:]
 #在这里去掉第一行名字

attribute_amount = np.size(Train_old,1)-1
Attri = np.ones(attribute_amount)


#我或许需要将所有attribute都变成0和1
attribute_0 = Train_old[0]
attribute_amount = np.size(Train_old,1)-1



# In[5]:


attribute_0


# In[6]:


Train = Train_old.copy()
for i in range(len(Train)):
    for k in range(attribute_amount):
        
        if Train[i,k] == attribute_0[k]:
            Train[i,k] = 0
        else:
            Train[i,k] = 1
            
Test = Test_old.copy()           
for i in range(len(Test)):
    for k in range(attribute_amount):
        if Test[i,k] == attribute_0[k]:
            Test[i,k] = 0
        else:
            Test[i,k] = 1


# In[7]:


label_type_A = Train[0,-1]
for i in range(len(Train)):
    if Train[i,-1] != label_type_A:
        label_type_B = Train[i,-1] 
        break

if label_type_A >= label_type_B:
    label_0 = label_type_A
    label_1 = label_type_B
else:
    label_0 = label_type_B
    label_1 = label_type_A


# In[8]:


label_set = np.array([label_0,label_1])


# In[10]:


for i in range(len(Train)):
    if Train[i,-1] == label_0:
        Train[i,-1] = 0
    else:
        Train[i,-1] = 1


# In[11]:


for i in range(len(Test)):
    if Test[i,-1] == label_0:
        Test[i,-1] = 0
    else:
        Test[i,-1] = 1


# In[12]:


Train = Train.astype(int)


# In[13]:


Test = Test.astype(int)


# ####################################################

# In[14]:


def gini(dataset):
    sum = len(dataset)
    
    if sum == 0:
        return 0
    
    else:
        
        A = 0      #######################################
        A_count = 0
        for i in range(sum):
            if dataset[i,-1] == A:
                A_count += 1
        Gini = 1 - (A_count/sum)**2 - ((sum - A_count)/sum)**2
        return Gini


# In[15]:


def count_attribute(Attribute):
    count = 0                        # 1 表示有， 0 表示没有
    for i in range(len(Attribute)):
        if Attribute[i] == 1:
            count += 1
    return count


# In[16]:


def attribute_number_set(Attribute):
    result = []
    for i in range(len(Attribute)):
        if Attribute[i] == 1:
            result.append(i)           ############
            
    result = np.array(result)               ###################3

    return result


# In[17]:


def generate_sub(dataset,attribute_number): #attribute_number = 0,1,2,3...
    A = 0                     ##########################3
    dataset_sub_one = []
    dataset_sub_two = []

    for i in range(len(dataset)):
        if dataset[i,int(attribute_number)] == A:      #0 为 左， 1 为 右
            dataset_sub_one.append(dataset[i])
        else:
            dataset_sub_two.append(dataset[i])

    dataset_sub_one = np.array(dataset_sub_one)   #list 变为 array
    dataset_sub_two = np.array(dataset_sub_two)

    return [dataset_sub_one,dataset_sub_two]


# In[18]:


def gini_gain(dataset,dataset_sub_one,dataset_sub_two):
    gain = gini(dataset) - (len(dataset_sub_one)/len(dataset))*gini(dataset_sub_one) - (len(dataset_sub_two)/len(dataset))*gini(dataset_sub_two)
    return gain


# In[19]:


def major(dataset):     # find the major label
    label_type_A = 0   ########################
    label_type_B = 1
    amount_A = 0

    for i in range(len(dataset)):
        if dataset[i,-1] == label_type_A:
            amount_A += 1
    amount_B = len(dataset) - amount_A
    if amount_A >= amount_B:
        value = label_type_A
    else:
        value = label_type_B

    return value


# #########################################

# In[20]:


class Node:
    def __init__(self,key):
        self.left = None
        self.right = None
        self.val = key
        self.A = None


# In[21]:


def Tree(Attribute,dataset,depth):   #此处的attribute为[1,1,1,1,1...1] , dataset 为 train的, depth 初始定位0
    
    node_instance = Node(key=-1)

    gini_impurity = gini(dataset)
    
    Attribute_number_set = attribute_number_set(Attribute)
    
    if depth == depth_max: #输入的全局变量
        
        node_instance.val = major(dataset)
        
        
            
        print('|'*depth,f'{label_set[node_instance.val]}',len(dataset))
        
        return node_instance

    else:
        if gini_impurity == 0:
            
            node_instance.val = dataset[0,-1]  #完全纯净，所有的label都一样
            
            print('|'*depth,f'{label_set[node_instance.val]}',len(dataset))
            
            return node_instance

        elif count_attribute(Attribute) == 0:
            node_instance.val = major(dataset)
            
            print('|'*depth,f'{label_set[node_instance.val]}',len(dataset))
            
            return node_instance

        else:
  
            
            
            Gini_gain = 0
            for i in Attribute_number_set:
                sub = generate_sub(dataset,i)                #0为左， 1为右
                sub_one = sub[0]                             
                sub_two = sub[1]
                if gini_gain(dataset,sub_one,sub_two) > Gini_gain:
                    Gini_gain = gini_gain(dataset,sub_one,sub_two)
                    attribute_target = i
                    sub_one_target = sub_one
                    sub_two_target = sub_two
                    
            if Gini_gain == 0:
                node_instance.val = major(dataset)
                
                print('|'*depth,f'{label_set[node_instance.val]}',len(dataset))
                
                return node_instance
                
            if Gini_gain > 0:
               
                    
                    
                    #
                    #node_instance.A = 1

                node_instance.val = attribute_target

                print('|'*depth,attribute_name[attribute_target],len(sub_one_target),'/',len(sub_two_target))   
                
                Attribute[int(attribute_target)] = 0            ################################################3
                Attribute_new_1 = Attribute.copy()
                Attribute_new_2 = Attribute.copy()
                

                node_instance.left = Tree(Attribute_new_1,sub_one_target,depth + 1)
                node_instance.right = Tree(Attribute_new_2,sub_two_target,depth + 1)

                return node_instance


# In[22]:


def print_tree(node, depth=0):
    if node == None: return
    depth += 1
    print((depth-1)*"|", node.val)
    #print(node.A)
    print_tree(node.left, depth)
    print_tree(node.right, depth)
#     print(node.val, depth)
   


# In[23]:


def find(val,node_find):   #val 是输入的test的attribute 数组（单个行）

    if node_find.left == None:
        return node_find.val

    elif val[int(node_find.val)] == 0:
        return find(val,node_find.left)

    elif val[int(node_find.val)] == 1:
        return find(val,node_find.right)


# In[24]:


def label_generate(Attribute,train_dataset,test_dataset,depth): # test dataset
    tmp_node = Tree(Attribute,train_dataset,depth)
    #print_tree(tmp_node)
    predict_set =np.ones(len(test_dataset))

    for i in range(len(test_dataset)):
       
        predict_set[i] = find(test_dataset[i],tmp_node) #需要去掉最后一个元素（label),用吗？
#         print(i, find(test_dataset[i],tmp_node))

    return predict_set


# In[25]:


Train_predict = label_generate(Attri,Train,Train,Depth)

Attri = np.ones(attribute_amount)########
Depth =0
Test_predict = label_generate(Attri,Train,Test,Depth)


# In[26]:


Train_predict_new = np.empty(len(Train),dtype = '<U25')

for i in range(len(Train)):
    if Train_predict[i] == 0:
        Train_predict_new[i] = label_0
    else:
        Train_predict_new[i] = label_1


# In[ ]:


Test_predict_new = np.empty(len(Test),dtype = '<U25')

for i in range(len(Test)):
    if Test_predict[i] == 0:
        Test_predict_new[i] = label_0
    else:
        Test_predict_new[i] = label_1


# In[27]:


train_error = 0
for i in range(len(Train)):
    if Train_predict[i] != Train[i,-1]:
        train_error += 1
train_error_rate = train_error/len(Train)
        
test_error = 0
for i in range(len(Test)):
    if Test_predict[i] != Test[i,-1]:
        test_error += 1
test_error_rate = test_error/len(Test)


# In[ ]:


f=open(train_output,'w')

for i in range(len(Train_predict_new)):
    f.write(f'{Train_predict_new[i]}')
    f.write('\n')
f.close()
        
f=open(test_output,'w')
for i in range(len(Test_predict_new)):
    f.write(f'{Test_predict_new[i]}')
    f.write('\n')
f.close()

                
f=open(metrics,'w')
f.write(f'error(train): {train_error_rate}\nerror(test): {test_error_rate}')
f.close()


# In[ ]:




