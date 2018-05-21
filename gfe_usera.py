
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


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

def load_a_cond_data(gfe_path=GFE_PATH):
    csv_pathc = os.path.join(gfe_path, "a_conditional_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathc)
def load_a_cond_target(gfe_path=GFE_PATH):
    csv_targetc = os.path.join(gfe_path, "a_conditional_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetc)

def load_a_doubtq_data(gfe_path=GFE_PATH):
    csv_pathd = os.path.join(gfe_path, "a_doubt_question_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathd)
def load_a_doubtq_target(gfe_path=GFE_PATH):
    csv_targetd = os.path.join(gfe_path, "a_doubts_question_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetd)

def load_a_emphasis_data(gfe_path=GFE_PATH):
    csv_pathe = os.path.join(gfe_path, "a_emphasis_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathe)
def load_a_emphasis_target(gfe_path=GFE_PATH):
    csv_targete = os.path.join(gfe_path, "a_emphasis_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targete)

def load_a_neg_data(gfe_path=GFE_PATH):
    csv_pathn = os.path.join(gfe_path, "a_negative_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathn)
def load_a_neg_target(gfe_path=GFE_PATH):
    csv_targetn = os.path.join(gfe_path, "a_negative_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetn)

def load_a_rel_data(gfe_path=GFE_PATH):
    csv_pathr = os.path.join(gfe_path, "a_relative_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathr)
def load_a_rel_target(gfe_path=GFE_PATH):
    csv_targetr = os.path.join(gfe_path, "a_relative_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetr)

def load_a_topics_data(gfe_path=GFE_PATH):
    csv_patht = os.path.join(gfe_path, "a_topics_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_patht)
def load_a_topics_target(gfe_path=GFE_PATH):
    csv_targett = os.path.join(gfe_path, "a_topics_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targett)

def load_a_wh_data(gfe_path=GFE_PATH):
    csv_pathw = os.path.join(gfe_path, "a_wh_question_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathw)
def load_a_wh_target(gfe_path=GFE_PATH):
    csv_targetw = os.path.join(gfe_path, "a_wh_question_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetw)

def load_a_yn_data(gfe_path=GFE_PATH):
    csv_pathy = os.path.join(gfe_path, "a_yn_question_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathy)
def load_a_yn_target(gfe_path=GFE_PATH):
    csv_targety = os.path.join(gfe_path, "a_yn_question_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targety)


# In[3]:


affirmd = load_a_affirm_data()
affirmd.head()
affirmt = load_a_affirm_target()
affirmt.head()

condd = load_a_cond_data()
condd.head()
condt = load_a_cond_target()
condt.head()

doubtqd = load_a_doubtq_data()
doubtqd.head()
doubtqt = load_a_doubtq_target()
doubtqt.head()

emphd = load_a_emphasis_data()
emphd.head()
empht = load_a_emphasis_target()
empht.head()

negd = load_a_neg_data()
negd.head()
negt = load_a_neg_target()
negt.head()

reld = load_a_rel_data()
reld.head()
relt = load_a_rel_target()
relt.head()

topicsd = load_a_topics_data()
topicsd.head()
topicst = load_a_topics_target()
topicst.head()

whd = load_a_wh_data()
whd.head()
wht = load_a_wh_target()
wht.head()

ynd = load_a_yn_data()
ynd.head()
ynt = load_a_yn_target()
ynt.head()


# In[4]:


qs_combine = pd.concat([affirmd,condd,doubtqd,emphd,negd,reld,topicsd,whd,ynd],ignore_index=True)


# In[5]:


len(qs_combine)


# In[6]:


qs_combine.info()
qs_combine.head()
qs_combine.drop('0.0',axis=1)      #droppinng the time stamp


# In[7]:


qs_combine.describe()


# In[8]:


y = pd.concat([affirmt,condt,doubtqt,empht,negt,relt,topicst,wht,ynt],ignore_index=True)


# In[9]:


y.info()
y.head()
len(y)


# In[10]:


y.drop([10])
y.info()
len(y)
y.head()
y.describe()


# In[11]:


qs_combine['y']=y


# In[12]:


from sklearn.model_selection import train_test_split
y=qs_combine['y']
X_train,X_test,y_train,y_test = train_test_split(qs_combine.iloc[:,1:],y,stratify=y)


# In[13]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)


# In[14]:


clf.score(X_train, y_train)


# In[15]:


clf.fit(X_test, y_test)


# In[16]:


clf.score(X_test, y_test)


# In[17]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)


# In[18]:


clf.fit(X_train,y_train)


# In[19]:


clf.score(X_train,y_train)


# In[20]:


clf.score(X_test,y_test)


# In[21]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[22]:


knn.score(X_train,y_train)


# In[23]:


knn.score(X_test,y_test)


# In[24]:


from sklearn.preprocessing import scale
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_clf = LDA(solver='lsqr',store_covariance=True)
lda_clf.fit(X_train,y_train)

print('Class Priors =', lda_clf.priors_)
print('Class Means =', lda_clf.means_[0], lda_clf.means_[1])
print('Coeffecients =', lda_clf.coef_)



# In[25]:


y_predicted = lda_clf.predict(X_test)
print('\n The error rate of the LDA model is {0:.2f}% '.format(100*np.mean(y_predicted!=y_test)))


# In[26]:


qs_combine.drop(['36x', '36y', '36z', '37x', '37y', '37z', '38x', '38y', '38z', '39x', '39y', '39z', '40x', '40y', '40z', '41x', '41y', '41z', '42x', '42y', '42z', '43x', '43y', '43z', '44x', '44y', '44z', '45x', '45y', '45z', '46x', '46y', '46z', '47x', '47y', '47z'], axis=1, inplace=True)


# In[27]:


from sklearn.model_selection import train_test_split
y=qs_combine['y']
X_train,X_test,y_train,y_test = train_test_split(qs_combine.iloc[:,1:],y,stratify=y)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)


# In[28]:


clf.score(X_train, y_train)


# In[29]:


clf.score(X_test, y_test)


# In[39]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)


# In[40]:


clf.fit(X_train,y_train)


# In[41]:


clf.score(X_train,y_train)


# In[42]:


clf.score(X_test,y_test)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[35]:


knn.score(X_train,y_train)


# In[36]:


knn.score(X_test,y_test)


# In[37]:


from sklearn.preprocessing import scale
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_clf = LDA(solver='lsqr',store_covariance=True)
lda_clf.fit(X_train,y_train)

print('Class Priors =', lda_clf.priors_)
print('Class Means =', lda_clf.means_[0], lda_clf.means_[1])
print('Coeffecients =', lda_clf.coef_)


# In[38]:


y_predicted = lda_clf.predict(X_test)
print('\n The error rate of the LDA model is {0:.2f}% '.format(100*np.mean(y_predicted!=y_test)))

