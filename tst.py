
# coding: utf-8

# In[3]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:

s = pd.Series([1, 3, 5, np.nan, 6, 8])


# In[6]:

s


# In[7]:

dates = pd.date_range('20130101', periods=6)


# In[9]:

dates


# In[10]:

df = pd.DataFrame(np.random.randn(6,4), index=dates, columns=list('ABCD'))


# In[13]:

df


# In[18]:

df2 = pd.DataFrame({'A': 1.,
                   'B': pd.Timestamp('20130102'),
                  'C': pd.Series(1, index=list(range(4)), dtype='float32'), 
                  'D': np.array([3]*4, dtype='int32'),
                  'E': pd.Categorical(["test", "train", "test", "train"]),
                  'F': 'foo'})


# In[19]:

df2


# In[20]:

df2.dtypes


# In[22]:

df.head()


# In[23]:

df.tail(3)


# In[24]:

df.index


# In[25]:

df.columns


# In[27]:

df.values


# In[29]:

df.describe()


# In[30]:

df.T


# In[31]:

df.sort_index(axis=1, ascending=False)


# In[33]:

df.sort(columns = 'B')


# In[35]:

df['A']


# In[37]:

df[0:3]


# In[41]:

df.loc[:,['A','B']]


# In[43]:

df.loc[dates[0]]


# In[45]:

df.loc['20130102':'20130104', ['A', 'B']]


# In[47]:

df.loc['20130101', 'A']


# In[49]:

df.iloc[3]


# In[51]:

df.iloc[3:5, 0:2]


# In[52]:

df.iloc[1:3, :]


# In[54]:

df.iloc[:, 1:3]


# In[58]:

df.iloc[1,1]


# In[60]:

df.iat[1,1]


# In[63]:

df[df.A > 1]


# In[65]:

df[df > 1]


# In[67]:

df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']


# In[69]:

df2


# In[71]:

df2[df2['E'].isin(['two', 'four'])]


# In[73]:

s1 = pd.Series([1,2,3,4,5,6], index=pd.date_range('20130102', periods=6))


# In[75]:

s1


# In[77]:

df['F'] = s1


# In[79]:

df.at[dates[0], 'A'] = 0


# In[81]:

df


# In[83]:

df.loc[:, 'D'] = np.array([5]*len(df))


# In[85]:

df


# In[87]:

df2 = df.copy()


# In[89]:

df2[df2 > 1] = -df2


# In[91]:

df2


# In[93]:

df1 = df.reindex(index=dates[0:4], columns=list(df.columns) + ['E'])


# In[95]:

df1.loc[dates[0]:dates[1], 'E'] = 1


# In[97]:

df1


# In[99]:

df1.dropna(how='any')


# In[101]:

df1.fillna(value=5)


# In[106]:

pd.isnull(df1)


# In[109]:

df.mean(1)


# In[111]:

get_ipython().magic('matplotlib inline')


# In[113]:

import matplotlib
import matplotlib.pyplot as plt


# In[115]:

import numpy as np


# In[117]:

from pylab import *


# In[119]:

x = np.linspace(0, 5, 10)
y = x ** 2


# In[122]:

figure() 
plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('tst')
show()


# In[134]:

fig = plt.figure(figsize = (12, 3))
axes = fig.add_axes([0.1, 0.1, 0.8, 0.8])
axes.plot(x, y, 'r')

axes.set_xlabel('x')
axes.set_ylabel('y')
axes.set_title('title');


# In[133]:

fig, axes = plt.subplots(nrows = 1, ncols=3)

for ax in axes:
    ax.plot(x, y, 'r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('title')


# In[ ]:



