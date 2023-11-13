#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:/Users/Rajesh Gonnade/Downloads/Titanic-Dataset.csv")
df


# In[3]:


df.describe()


# In[4]:


df.info()


# In[5]:


df.shape


# In[6]:


df.nunique()


# In[7]:


df.isnull().sum()


# In[8]:


mean=df['Age'].mean()
df['Age']=df['Age'].replace(np.nan,mean)
df.isnull().sum()


# In[9]:


df['Age'].fillna(mean,inplace=True)
df


# In[10]:


mode=df['Embarked'].mode()
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])
df.isnull().sum()


# In[11]:


df.drop('Cabin', axis=1, inplace=True)
df


# In[12]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
corr = df[numerical_columns].corr()

corr=df.corr()
plt.figure(figsize=(7, 5))
sns.heatmap(corr,annot=True,fmt='.2f')
plt.show()


# In[13]:


sns.countplot(x=df['Survived'])
plt.show()
dead=round(df['Survived'].value_counts().values[0])
print("Out of 891, {} people died in the accident".format(dead))


# In[14]:


plt.figure(figsize=(17,10))
plt.bar(df['Age'],df['Fare'],color='purple',linestyle='--')
plt.xlabel('Age',size='30')
plt.ylabel('Fare',size='30')
plt.title("Bar Plot",size='30')
plt.show()


# In[15]:


# Assuming you have a DataFrame named 'df' with columns 'Survived' and 'Age'
plt.figure(figsize=(17, 10))
sns.histplot(data=df, x='Age', hue='Survived', bins=30, color='green', linestyle='--')
plt.xlabel('Age', size=30)
plt.ylabel('Count', size=30)
plt.title('Distribution of Age among Survivors and Non-Survivors', size=30)
plt.legend(title='Survived', labels=['No', 'Yes'])
plt.show()


# In[16]:


print((df['Pclass'].value_counts()))


# In[17]:


# Assuming you have a DataFrame named 'df' with columns 'Survived' and 'Pclass'
plt.figure(figsize=(10, 6))
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title('Survival Distribution Across Passenger Classes')
plt.xlabel('Passenger Class')
plt.ylabel('Count')
plt.show()


# In[ ]:





# In[ ]:




