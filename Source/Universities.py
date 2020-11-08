#!/usr/bin/env python
# coding: utf-8

# In[24]:


#importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, classification_report

get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')


# In[25]:


#Importing the data
data = pd.read_csv("data.csv")
data.head()

#Checking the info() and describe() methods on the data
data.info()
data.describe()


# In[26]:


#data visualizations (EDA)
sns.scatterplot('room_board', 'grad_rate', data=data, hue='private')

sns.scatterplot('outstate', 'f_undergrad', data=data, hue='private')

plt.figure(figsize=(12, 8))

data.loc[data.private == 'Yes', 'outstate'].hist(label="Private College", bins=30)
data.loc[data.private == 'No', 'outstate'].hist(label="Non Private College", bins=30)

plt.xlabel('Outstate')
plt.legend()

plt.figure(figsize=(12, 8))

data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)

plt.xlabel('Graduation Rate')
plt.legend()

#A private school with a graduation rate of higher than 100, so adjust it.
data.loc[data.grad_rate > 100, 'grad_rate'] = 100

#revisualising it
data.loc[data.private == 'Yes', 'grad_rate'].hist(label="Private College", bins=30)
data.loc[data.private == 'No', 'grad_rate'].hist(label="Non Private College", bins=30)

plt.xlabel('Graduation Rate')
plt.legend()


# In[27]:


#applying k means to form cluster

#kmeans model with 2 cluster
kmeans = KMeans(2)

#applying k means
kmeans.fit(data.drop('private', axis=1))
kmeans.cluster_centers_


# In[29]:


#Evaluting result

data['Cluster'] = data['private'].apply(lambda x: 1 if x == 'Yes' else 0)
data.head()
print(confusion_matrix(data['Cluster'], kmeans.labels_))
print(classification_report(data['Cluster'], kmeans.labels_))

