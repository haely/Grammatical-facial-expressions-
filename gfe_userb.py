
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import os
GFE_PATH = "C:\Haely\MS2017\sem2\EE 259\Project\grammatical_facial_expression"

def load_b_affirm_data(gfe_path=GFE_PATH):
    csv_pathab = os.path.join(gfe_path, "b_affirmative_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathab)
def load_b_affirm_target(gfe_path=GFE_PATH):
    csv_targetab = os.path.join(gfe_path, "b_affirmative_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetab)

def load_b_cond_data(gfe_path=GFE_PATH):
    csv_pathcb = os.path.join(gfe_path, "b_conditional_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathcb)
def load_b_cond_target(gfe_path=GFE_PATH):
    csv_targetcb = os.path.join(gfe_path, "b_conditional_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetcb)

def load_b_doubtq_data(gfe_path=GFE_PATH):
    csv_pathdb = os.path.join(gfe_path, "b_doubt_question_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathdb)
def load_b_doubtq_target(gfe_path=GFE_PATH):
    csv_targetdb = os.path.join(gfe_path, "b_doubt_question_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetdb)

def load_b_emphasis_data(gfe_path=GFE_PATH):
    csv_patheb = os.path.join(gfe_path, "b_emphasis_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_patheb)
def load_b_emphasis_target(gfe_path=GFE_PATH):
    csv_targeteb = os.path.join(gfe_path, "b_emphasis_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targeteb)

def load_b_neg_data(gfe_path=GFE_PATH):
    csv_pathnb = os.path.join(gfe_path, "b_negative_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathnb)
def load_b_neg_target(gfe_path=GFE_PATH):
    csv_targetnb = os.path.join(gfe_path, "b_negative_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetnb)

def load_b_rel_data(gfe_path=GFE_PATH):
    csv_pathrb = os.path.join(gfe_path, "b_relative_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathrb)
def load_b_rel_target(gfe_path=GFE_PATH):
    csv_targetrb = os.path.join(gfe_path, "b_relative_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetrb)

def load_b_topics_data(gfe_path=GFE_PATH):
    csv_pathtb = os.path.join(gfe_path, "b_topics_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathtb)
def load_b_topics_target(gfe_path=GFE_PATH):
    csv_targettb = os.path.join(gfe_path, "b_topics_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targettb)

def load_b_wh_data(gfe_path=GFE_PATH):
    csv_pathwb = os.path.join(gfe_path, "b_wh_question_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathwb)
def load_b_wh_target(gfe_path=GFE_PATH):
    csv_targetwb = os.path.join(gfe_path, "b_wh_question_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetwb)

def load_b_yn_data(gfe_path=GFE_PATH):
    csv_pathyb = os.path.join(gfe_path, "b_yn_question_datapoints.csv")
    print(gfe_path)
    return pd.read_csv(csv_pathyb)
def load_b_yn_target(gfe_path=GFE_PATH):
    csv_targetyb = os.path.join(gfe_path, "b_yn_question_targets.csv")
    print(gfe_path)
    return pd.read_csv(csv_targetyb)


# In[3]:


affirmdb = load_b_affirm_data()
affirmdb.head()
affirmtb = load_b_affirm_target()
affirmtb.head()

conddb = load_b_cond_data()
conddb.head()
condtb = load_b_cond_target()
condtb.head()

doubtqdb = load_b_doubtq_data()
doubtqdb.head()
doubtqtb = load_b_doubtq_target()
doubtqtb.head()

emphdb = load_b_emphasis_data()
emphdb.head()
emphtb = load_b_emphasis_target()
emphtb.head()

negdb = load_b_neg_data()
negdb.head()
negtb = load_b_neg_target()
negtb.head()

reldb = load_b_rel_data()
reldb.head()
reltb = load_b_rel_target()
reltb.head()

topicsdb = load_b_topics_data()
topicsdb.head()
topicstb = load_b_topics_target()
topicstb.head()

whdb = load_b_wh_data()
whdb.head()
whtb = load_b_wh_target()
whtb.head()

yndb = load_b_yn_data()
yndb.head()
yntb = load_b_yn_target()
yntb.head()


# In[4]:


qs_combine = pd.concat([affirmdb,conddb,doubtqdb,emphdb,negdb,reldb,topicsdb,whdb,yndb],ignore_index=True)


# In[5]:


len(qs_combine)


# In[6]:


qs_combine.info()


# In[7]:


qs_combine.head()


# In[8]:


qs_combine.drop('0.0',axis=1)      #dropping the time stamp


# In[9]:


y = pd.concat([affirmtb,condtb,doubtqtb,emphtb,negtb,reltb,topicstb,whtb,yntb],ignore_index=True)


# In[10]:


y.info()
y.head()
len(y)


# In[11]:


qs_combine['y']=y
y.drop([10])        #dropping the row header: named it 10 in dataset


# # Developing various models with all 300 features

# In[12]:


from sklearn.model_selection import train_test_split
y=qs_combine['y']
X_train,X_test,y_train,y_test = train_test_split(qs_combine.iloc[:,1:],y,stratify=y)


# # Logistic regression

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


# # Random forest

# In[17]:


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50)


# In[18]:


clf.fit(X_train,y_train)


# In[19]:


clf.score(X_train,y_train)


# In[20]:


clf.score(X_test,y_test)


# # KNN

# In[21]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[22]:


knn.score(X_train,y_train)


# In[23]:


knn.score(X_test,y_test)


# # LDA 

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


# # Dropping nose co-ordinates and developing the same models

# In[26]:


qs_combine.drop(['36x', '36y', '36z', '37x', '37y', '37z', '38x', '38y', '38z', '39x', '39y', '39z', '40x', '40y', '40z', '41x', '41y', '41z', '42x', '42y', '42z', '43x', '43y', '43z', '44x', '44y', '44z', '45x', '45y', '45z', '46x', '46y', '46z', '47x', '47y', '47z'], axis=1, inplace=True)


# # Logistic regression

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


# # Random forest

# In[39]:




from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)


# In[40]:


clf.fit(X_train,y_train)


# In[41]:


clf.score(X_train,y_train)


# In[42]:


clf.score(X_test,y_test)


# # KNN

# In[34]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)


# In[35]:


knn.score(X_train,y_train)


# In[36]:


knn.score(X_test,y_test)


# # LDA

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


# # Cross validation score

# In[45]:


from sklearn.model_selection import cross_val_score
print("Cross validation score for Rf: ")
cross_val_score(clf22, X_train, y_train.reshape(len(y_train),), cv=3, scoring = "accuracy")

