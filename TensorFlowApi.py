#!/usr/bin/env python
# coding: utf-8

# In[5]:


import tensorflow as tf


# In[6]:


mnist = tf.keras.datasets.mnist


# In[7]:


(x_train, y_train),(x_test, y_test) = mnist.load_data()


# In[8]:


print(x_train[2])


# In[ ]:




