
# coding: utf-8

# In[53]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import axes3d
import seaborn as sns

from sklearn.preprocessing import scale


# In[45]:


import os
GFE_PATH = "C:\Haely\MS2017\sem2\EE 259\Project\grammatical_facial_expression"

def load_a_affirm_data(gfe_path=GFE_PATH):
    csv_patha = os.path.join(gfe_path, "a_affirmative_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_patha)
def load_a_affirm_target(gfe_path=GFE_PATH):
    csv_targeta = os.path.join(gfe_path, "a_affirmative_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targeta)


# In[46]:


affirmd = load_a_affirm_data()
affirmd.head()
affirmt = load_a_affirm_target()
affirmt.head()


# In[47]:


from statsmodels.graphics.regressionplots import abline_plot


# In[49]:


#plt.scatter(affirmd, affirmt)


# In[ ]:


affirmd['99z'].plot()


# In[ ]:


affirmt.plot()


# In[ ]:


import matplotlib.pyplot as plt
affirmd_drop=affirmd.drop('0.0',axis=1)
affirmd_drop.head(3)


# In[ ]:


affirmd_drop.shape


# In[ ]:


feature1 = affirmd_drop['0'].values
classe = affirmt['10'].values


# In[52]:


plt.plot(feature1)
plt.show()


# In[ ]:


plt.plot(classe)
plt.show()


# In[ ]:


plt.plot(feature1, classe)
plt.show()


# In[ ]:


affirm_all=pd.concat([affirmd, affirmt], axis=1)
affirm_all.head()
affirm_all_drop=affirm_all.drop('0.0',axis=1)
affirm_all_drop.head(3)


# In[ ]:


import pandas as pd
import numpy as np

#df = pd.DataFrame(np.random.randn(100, 6), columns=['a', 'b', 'c', 'd', 'e', 'f'])

ax = affirm_all_drop.plot(kind="scatter", x="55z",y="10", color="c", label="a vs. x")
affirm_all_drop.plot(kind='scatter', x='0x', y='10', color='r',ax=ax)    
affirm_all_drop.plot(kind='scatter', x='0y', y='10', color='g', ax=ax)    
#affirm_all_drop.plot(kind='scatter', x='0z', y='10', color='b', ax=ax)
plt.show()
affirm_all_drop.plot(kind='scatter', x='90x', y='10', color='c',ax=ax)    
affirm_all_drop.plot(kind='scatter', x='90y', y='10', color='m', ax=ax)    
affirm_all_drop.plot(kind='scatter', x='90y', y='10', color='y', ax=ax)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import pandas
import numpy
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
#names = ['4x', '4y', '4z', '12x', '12y', '12z', '41x', '41y', '41z', '64x', '64y', '64z']

correlations = affirmd.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
import pandas
import numpy
#url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
#names = ['4x', '4y', '4z', '12x', '12y', '12z', '41x', '41y', '41z', '64x', '64y', '64z']

correlations = affirmt.corr()
# plot correlation matrix
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = numpy.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
ax.set_xticklabels(names)
ax.set_yticklabels(names)
plt.show()

