#!/usr/bin/env python
# coding: utf-8

# In[105]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import string


# In[94]:


#reading the data
train_data = pd.read_csv('train.txt')
test_data = pd.read_csv('test.txt')
val_data = pd.read_csv('val.txt')


# In[95]:


#data processing and cleaning training data
train_data.shift(periods=1)[0] = 'i didnt feel humiliated;sadness'
train_data.rename(columns={'i didnt feel humiliated;sadness': 'Message'}, inplace = True)
train_data[['Message', 'Class']] = train_data.Message.str.split(";", expand = True)


# In[96]:


#data processing and cleaning testing data
test_data.shift(periods=1)[0] = 'im feeling rather rotten so im not very ambitious right now;sadness'
test_data.rename(columns={'im feeling rather rotten so im not very ambitious right now;sadness': 'Message'}, inplace = True)
test_data[['Message', 'Class']] = test_data.Message.str.split(";", expand = True)


# In[97]:


#data processing and cleaning vaidation data
val_data.shift(periods=1)[0] = 'im feeling quite sad and sorry for myself but ill snap out of it soon;sadness'
val_data.rename(columns={'im feeling quite sad and sorry for myself but ill snap out of it soon;sadness': 'Message'}, inplace = True)
val_data[['Message', 'Class']] = val_data.Message.str.split(";", expand = True)


# In[98]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
stop = stopwords.words('english')


# In[106]:


#the function to process the data
def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


# In[114]:


from sklearn.svm import SVC
model = SVC()


# In[115]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


# In[130]:


from sklearn.pipeline import Pipeline

#making a pipeline for processing the data
pipeline = Pipeline([
    ('bow', CountVectorizer(analyzer=text_process)),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', SVC()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])


# In[131]:


#fitting the data
pipeline.fit(train_data['Message'],train_data['Class'])


# In[149]:


#predicting on validation data
predictions = pipeline.predict(val_data['Message'])


# In[150]:


from sklearn.metrics import classification_report, confusion_matrix


# In[151]:


#printing the classification report for va_data
print(classification_report(predictions,val_data['Class']))


# In[153]:


#predicting and printing the the classification report on test_data
predictions = pipeline.predict(test_data['Message'])
print(classification_report(predictions,test_data['Class']))


# In[197]:


#example from test_data
message = test_data['Message'].iloc[249]
print('message:',message)
print("Emotion:", test_data['Class'].iloc[4])
print("Expected Emotion:", predictions[4])


# In[ ]:




