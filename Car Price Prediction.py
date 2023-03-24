#!/usr/bin/env python
# coding: utf-8

# # About this file 
# This dataset contains information about used cars listed on different websites. This data can be used for a lot of purposes such as price prediction to exemplify the use of linear regression in Machine Learning.
# 
# The columns in the given dataset is as follows :
# 
# 1. Car_Name
# 2. Year
# 3. Selling_Price
# 4. Present_Price
# 5. Kms_Driven
# 6. Fuel_Type
# 7. Seller_Type
# 8. Transmission
# 9. Owner

# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[19]:


# loading the csv file 
df = pd.read_csv('Car data.csv')


# In[20]:


df.head(10)


# In[21]:


df.shape


# In[22]:


df.info()


# In[23]:


df.describe()


# In[25]:


# checking the distribution of categorical data
print(df['Fuel_Type'].value_counts())
print(df['Seller_Type'].value_counts())
print(df['Transmission'].value_counts())


# **Encoding the Categorical Data**

# In[26]:


# encoding "Fuel_Type" Column
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
df.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[27]:


df.head()


# **Splitting the data into Train and Test**

# In[ ]:




