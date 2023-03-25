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

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics


# In[2]:


# loading the csv file 
df = pd.read_csv('Car data.csv')


# In[3]:


df.head(10)


# In[4]:


df.shape


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


plt.figure(figsize = (10,7))
sns.heatmap(df.corr(), annot = True)
plt.title('Correlation between colums')


# In[8]:


fig = plt.figure(figsize = (10, 5))
plt.title('Correlation between present price and selling price')
sns.regplot(x = 'Present_Price', y = 'Selling_Price', data = df)


# In[9]:


# checking the distribution of categorical data
print(df['Fuel_Type'].value_counts())
print(df['Seller_Type'].value_counts())
print(df['Transmission'].value_counts())


# **Encoding the Categorical Data**

# In[10]:


# encoding "Fuel_Type" Column
df.replace({'Fuel_Type':{'Petrol':0,'Diesel':1,'CNG':2}},inplace=True)

# encoding "Seller_Type" Column
df.replace({'Seller_Type':{'Dealer':0,'Individual':1}},inplace=True)

# encoding "Transmission" Column
df.replace({'Transmission':{'Manual':0,'Automatic':1}},inplace=True)


# In[11]:


df.head()


# **Splitting the data**

# In[12]:


X =  df.drop(['Car_Name', 'Selling_Price'], axis = 1)
Y = df['Selling_Price']


# In[13]:


X


# In[14]:


Y


# **Splitting the data into Train and Test**

# In[15]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 2)


# **Training Our Model**

# In[16]:


LR_model = LinearRegression()


# In[17]:


LR_model.fit(X_train, Y_train)


# **Model Evaluation**

# In[18]:


# prediction on training data
train_data_prediction = LR_model.predict(X_train)


# In[19]:


# Comparing predicted and target value by R square Error
error = metrics.r2_score(Y_train, train_data_prediction)


# In[20]:


error


# In[21]:


plt.scatter(Y_train, train_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")


# In[22]:


# prediction on test data
test_data_prediction = LR_model.predict(X_test)


# In[23]:


# Comparing predicted and target value by R square Error
error = metrics.r2_score(Y_test, test_data_prediction)


# In[24]:


error


# In[25]:


plt.scatter(Y_test, test_data_prediction)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")

