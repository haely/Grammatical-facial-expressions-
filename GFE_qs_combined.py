
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


# In[4]:


affirmda = load_a_affirm_data()
affirmta = load_a_affirm_target()

condda = load_a_cond_data()
condta = load_a_cond_target()

doubtqda = load_a_doubtq_data()
doubtqta = load_a_doubtq_target()

emphda = load_a_emphasis_data()
emphta = load_a_emphasis_target()

negda = load_a_neg_data()
negta = load_a_neg_target()

relda = load_a_rel_data()
relta = load_a_rel_target()

topicsda = load_a_topics_data()
topicsta = load_a_topics_target()

whda = load_a_wh_data()
whta = load_a_wh_target()

ynda = load_a_yn_data()
ynta = load_a_yn_target()


# In[5]:


affirmdb = load_b_affirm_data()
affirmtb = load_b_affirm_target()

conddb = load_b_cond_data()
condtb = load_b_cond_target()

doubtqdb = load_b_doubtq_data()
doubtqtb = load_b_doubtq_target()

emphdb = load_b_emphasis_data()
emphtb = load_b_emphasis_target()

negdb = load_b_neg_data()
negtb = load_b_neg_target()

reldb = load_b_rel_data()
reltb = load_b_rel_target()

topicsdb = load_b_topics_data()
topicstb = load_b_topics_target()

whdb = load_b_wh_data()
whtb = load_b_wh_target()

yndb = load_b_yn_data()
yntb = load_b_yn_target()


# In[8]:


users_combine_affirmd = pd.concat([affirmda, affirmdb],ignore_index=True)
affirm_y = pd.concat([affirmta,affirmtb],ignore_index=True)

users_combine_condd = pd.concat([condda, conddb],ignore_index=True)
cond_y = pd.concat([condta, condtb],ignore_index=True)

users_combine_doubtqd = pd.concat([doubtqda, doubtqdb],ignore_index=True)
doubtq_y = pd.concat([doubtqta, doubtqtb],ignore_index=True)

users_combine_emphd = pd.concat([emphda, emphdb],ignore_index=True)
emph_y = pd.concat([emphta, emphtb],ignore_index=True)

users_combine_negd = pd.concat([negda, negdb],ignore_index=True)
neg_y = pd.concat([negta, negtb],ignore_index=True)

users_combine_reld = pd.concat([relda, reldb],ignore_index=True)
rel_y = pd.concat([relta, reltb],ignore_index=True)

users_combine_topicsd = pd.concat([topicsda, topicsdb],ignore_index=True)
topics_y = pd.concat([topicsta, topicstb],ignore_index=True)

users_combine_whd = pd.concat([whda, whdb],ignore_index=True)
wh_y = pd.concat([whta, whtb],ignore_index=True)

users_combine_ynd = pd.concat([ynda, yndb],ignore_index=True)
yn_y = pd.concat([ynta, yntb],ignore_index=True)


# In[11]:


users_combine_affirmd['affirm_y']=affirm_y
affirm_y.drop([10]) 



# In[12]:


users_combine_condd['cond_y']=cond_y
cond_y.drop([10]) 


# In[13]:


users_combine_doubtqd['doubtq_y']=doubtq_y
doubtq_y.drop([10]) 


# In[14]:


users_combine_emphd['emph_y']=emph_y
emph_y.drop([10]) 


# In[15]:


users_combine_negd['neg_y']=neg_y
neg_y.drop([10]) 


# In[16]:


users_combine_reld['rel_y']=rel_y
rel_y.drop([10]) 


# In[17]:


users_combine_topicsd['topics_y']=topics_y
topics_y.drop([10]) 


# In[18]:


users_combine_whd['wh_y']=wh_y
wh_y.drop([10]) 


# In[19]:


users_combine_ynd['yn_y']=yn_y
yn_y.drop([10]) 


# In[22]:


from sklearn.model_selection import train_test_split
ya=users_combine_affirmd['affirm_y']
Xa_train,Xa_test,ya_train,ya_test = train_test_split(users_combine_affirmd.iloc[:,1:],ya,stratify=ya)

yc=users_combine_condd['cond_y']
Xc_train,Xc_test,yc_train,yc_test = train_test_split(users_combine_condd.iloc[:,1:],yc,stratify=yc)

yd=users_combine_doubtqd['doubtq_y']
Xd_train,Xd_test,yd_train,yd_test = train_test_split(users_combine_doubtqd.iloc[:,1:],yd,stratify=yd)

ye=users_combine_emphd['emph_y']
Xe_train,Xe_test,ye_train,ye_test = train_test_split(users_combine_emphd.iloc[:,1:],ye,stratify=ye)

yn=users_combine_negd['neg_y']
Xn_train,Xn_test,yn_train,yn_test = train_test_split(users_combine_negd.iloc[:,1:],yn,stratify=yn)

yr=users_combine_reld['rel_y']
Xr_train,Xr_test,yr_train,yr_test = train_test_split(users_combine_reld.iloc[:,1:],yr,stratify=yr)

yt=users_combine_topicsd['topics_y']
Xt_train,Xt_test,yt_train,yt_test = train_test_split(users_combine_topicsd.iloc[:,1:],yt,stratify=yt)

yw=users_combine_whd['wh_y']
Xw_train,Xw_test,yw_train,yw_test = train_test_split(users_combine_whd.iloc[:,1:],yw,stratify=yw)

yy=users_combine_ynd['yn_y']
Xy_train,Xy_test,yy_train,yy_test = train_test_split(users_combine_ynd.iloc[:,1:],yy,stratify=yy)



# In[25]:


from sklearn.preprocessing import scale
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda_clf = LDA(solver='lsqr',store_covariance=True)

lda_clf.fit(Xa_train,ya_train)
ya_predicted = lda_clf.predict(Xa_test)
print('\n The error rate of the LDA model for affirm is {0:.2f}% '.format(100*np.mean(ya_predicted!=ya_test)))

lda_clf.fit(Xc_train,yc_train)
yc_predicted = lda_clf.predict(Xc_test)
print('\n The error rate of the LDA model for conditional is {0:.2f}% '.format(100*np.mean(yc_predicted!=yc_test)))

lda_clf.fit(Xd_train,yd_train)
yd_predicted = lda_clf.predict(Xd_test)
print('\n The error rate of the LDA model for doubt questions is {0:.2f}% '.format(100*np.mean(yd_predicted!=yd_test)))

lda_clf.fit(Xe_train,ye_train)
ye_predicted = lda_clf.predict(Xe_test)
print('\n The error rate of the LDA model for emphasis is {0:.2f}% '.format(100*np.mean(ye_predicted!=ye_test)))

lda_clf.fit(Xn_train,yn_train)
yn_predicted = lda_clf.predict(Xn_test)
print('\n The error rate of the LDA model for negative is {0:.2f}% '.format(100*np.mean(yn_predicted!=yn_test)))

lda_clf.fit(Xr_train,yr_train)
yr_predicted = lda_clf.predict(Xr_test)
print('\n The error rate of the LDA model for relativr is {0:.2f}% '.format(100*np.mean(yr_predicted!=yr_test)))

lda_clf.fit(Xt_train,yt_train)
yt_predicted = lda_clf.predict(Xt_test)
print('\n The error rate of the LDA model for topics is {0:.2f}% '.format(100*np.mean(yt_predicted!=yt_test)))

lda_clf.fit(Xw_train,yw_train)
yw_predicted = lda_clf.predict(Xw_test)
print('\n The error rate of the LDA model for wh questions is {0:.2f}% '.format(100*np.mean(yw_predicted!=yw_test)))

lda_clf.fit(Xy_train,yy_train)
yy_predicted = lda_clf.predict(Xy_test)
print('\n The error rate of the LDA model for yes or no is {0:.2f}% '.format(100*np.mean(yy_predicted!=yy_test)))

