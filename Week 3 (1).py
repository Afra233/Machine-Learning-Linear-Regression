#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.formula.api as smf


# In[32]:


credit = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\2nd Semester\\Credit.csv")


# In[33]:


credit.head()


# In[34]:


credit = pd.get_dummies(credit, columns=['Own', 'Student', 'Married', 'Region'], dtype=int, drop_first=True)
credit.head(3)


# In[35]:


X = credit['Own_Yes']
Y = credit['Balance']


# In[36]:


model5 = smf.ols(formula='Balance ~ Own_Yes', data=credit).fit()
model5.params


# In[39]:


X = credit[['Region_South','Region_West']]
Y = credit['Balance']


# In[43]:


model6 = smf.ols(formula='Balance ~ Region_South + Region_West + Income + Rating',
data=credit).fit()
model6.params


# In[44]:


model6.summary()


# In[45]:


df=pd.read_csv ("C:\\Users\\LENOVO\\Desktop\\2nd Semester\\Advertising.csv")


# In[47]:


model7 = smf.ols(formula='Sales ~ TV+Radio + TV*Radio', data=df).fit()
print(model7.summary())


# In[48]:


model8 = smf.ols(formula='Sales ~ TV+Radio', data=df).fit()
print(model8.summary())


# In[50]:


X = credit[['Income','Student_Yes']]
Y = credit['Balance']


# In[51]:


model7 = smf.ols(formula='Balance ~ Income + Student_Yes+Income * Student_Yes', data = credit).fit()
model7.params


# In[52]:


model7.summary()


# In[53]:


auto=pd.read_csv ("C:\\Users\\LENOVO\\Desktop\\2nd Semester\\Auto_clean.csv")


# In[54]:


X = auto['horsepower']
Y = auto['mpg']


# In[56]:


auto['horsepower'] = pd.to_numeric(auto['horsepower'], errors='coerce')
auto = auto.dropna(subset=['horsepower'])
model8 = smf.ols(formula='mpg ~ horsepower + I(horsepower**2)', data=auto).fit()
print(model8.summary())


# In[57]:


df.corr()


# In[58]:


import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm     
import statsmodels.formula.api as smf   


# In[59]:


from sklearn.model_selection import train_test_split


# In[61]:


from statsmodels.tools.eval_measures import rmse
auto = pd.read_csv("C:\\Users\\LENOVO\\Desktop\\2nd Semester\\Auto_clean.csv")


# In[62]:


train, test = train_test_split(auto, test_size = 196, random_state = 0)


# In[64]:


train


# In[65]:


model1_train = smf.ols(formula='mpg ~ horsepower', data= train).fit() 
print(model1_train.summary())


# In[69]:


train


# In[70]:


test


# In[66]:


model1_train_pred = model1_train.predict(train)
model1_test_pred = model1_train.predict(test)


# In[71]:


rmse_train = rmse(train.mpg,model1_train_pred)**2 
rmse_train


# In[72]:


rmse_test = rmse(test.mpg,model1_test_pred)**2
rmse_test


# In[ ]:




