
# coding: utf-8

# # Task6: Predicting Real Estate House Prices

# ## This task is provided to test your understanding of building a Linear Regression model for a provided dataset

# ### Dataset: Real_estate.csv

# ### Import the necessary libraries
# #### Hint: Also import seaborn

# In[2]:


import seaborn as sns
import pandas as pd 


# ### Read the csv data into a pandas dataframe and display the first 5 samples

# In[3]:


d=pd.read_csv('Real estate.csv')
d.head()


# ### Show more information about the dataset

# In[4]:


d.info()


# ### Find how many samples are there and how many columns are there in the dataset

# In[5]:


d.shape


# ### What are the features available in the dataset?

# In[6]:


d.columns


# ### Check if any features have missing data

# In[7]:


len(d) - d.count()


# ### Group all the features as dependent features in X

# In[8]:


X=d.iloc[:,:-1]
X


# ### Group feature(s) as independent features in y

# In[27]:


y = d.iloc[:,7]
y


# ### Split the dataset into train and test data

# In[49]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 8)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### Choose the model (Linear Regression)

# In[45]:


from sklearn.linear_model import LinearRegression   #importing


# ### Create an Estimator object

# In[62]:


linear_reg = LinearRegression()


# ### Train the model

# In[63]:


linear_reg.fit(X_train, y_train)


# ### Apply the model

# In[70]:



yp = linear_reg.predict(X_test)
yp


# ### Display the coefficients

# In[71]:


linear_reg.coef_


# ### Find how well the trained model did with testing data

# In[72]:


from sklearn.metrics import r2_score
print('r2 Score : ', r2_score(y_test, yp))


# ### Plot House Age Vs Price
# #### Hint: Use regplot in sns

# In[68]:


d_plot=sns.regplot(x="X2 house age", y="Y house price of unit area", data=d);
d_plot


# ### Plot Distance to MRT station Vs Price

# In[41]:


dist1 = sns.regplot(y="X3 distance to the nearest MRT station", x="Y house price of unit area", data=d)
dist1


# ### Plot Number of Convienience Stores Vs Price

# In[42]:


dist2 = sns.regplot(y="X4 number of convenience stores", x="Y house price of unit area", data=d)
dist2

