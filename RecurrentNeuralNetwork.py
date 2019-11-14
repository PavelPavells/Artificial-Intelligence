#!/usr/bin/env python
# coding: utf-8

# In[10]:


import tensorflow as tf
tf.enable_eager_execution()
import numpy as np
import os
import time
import functools


# In[12]:


#Dataset prepare
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
char2idx = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])
char_dataset = tf.data.Dataset.form_tensor_slices(text_as_int)


# In[13]:


#Constants
seq_length = 100
examples_per_epoch = len(text) // seq_length
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch // BATCH_SIZE
BUFFER_SIZE = 10000
embedding_dim = 256
rnn_units = 1024
vocab_size = len(vocab)
EPOCHS = 1


# In[20]:


#Functions
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, recurrent_initializer='glorot_uniform', stateful=True, recurrent_activation='sigmoid'),
        tf.keras.layers.Dense(vocab-size)
    ])
    return model
def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
def generate_text(model ,start_string):
    num_generate = 1000
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    temperature = 1.0
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predictions = predictions / temperature
        predicted_id = tf.multinominal(predictions, num_samples=1)[-1,0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text.generated.append(idx2char[predicated_id])
    return (start_string + ''.join(text.generated))
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text


# In[15]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
dataset = sequences.map(split_input_target)
dataset = sequences.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# In[16]:


model = build_model(
    vocab_size = len(vocab),
    embedding_dim = embedding_dim,
    rnn_units = rnn_units,
    batch_size = BATCH_SIZE)


# In[17]:


#Compile
model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)


# In[18]:


history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epochs=steps_per_epochs)


# In[19]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.build(tf.TensorShape([1, None]))


# In[21]:


print(generate_text(model, start_string=u"ROMEO: "))


# In[ ]:




