#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df= pd.read_csv('airlines.csv')


# In[3]:


df.head()


# In[4]:


len(df)


# In[5]:


df.isnull().sum()


# In[6]:


df.dtypes


# In[7]:


df.drop('id',axis=1,inplace=True)


# In[8]:


df


# In[9]:


cols= df.columns


# In[10]:


for col in cols:
    print(col,'\n')
    print(df[col].value_counts(),'\n')
    print('***********************','\n')


# In[11]:


from sklearn.preprocessing import OneHotEncoder
ohc= OneHotEncoder()


# In[12]:


X= ohc.fit_transform(df.Airline.values.reshape(-1,1)).toarray()


# In[13]:


Y= ohc.fit_transform(df.AirportFrom.values.reshape(-1,1)).toarray()
Z= ohc.fit_transform(df.AirportTo.values.reshape(-1,1)).toarray() 


# Airlines are OneHotEncoded
# AirportFrom and Airport to are LabelEncoded

# In[14]:


X


# In[15]:


np.unique(X)


# In[16]:


X.shape


# In[17]:


for i in [X,Y,Z]:
    print(i.shape)


# In[18]:


X1= pd.DataFrame(X,columns=["Airline_"+str(int(i)) for i in range(X.shape[1])])


# In[19]:


X1.head()


# In[20]:


df.shape


# In[21]:


copydf= df


# In[22]:


df= pd.concat([df,X1],axis=1)


# In[23]:


df.drop('Airline',axis=1,inplace=True)


# In[24]:


df.drop('Airline_0',axis=1,inplace=True)


# In[25]:


from sklearn.preprocessing import LabelEncoder
lc=LabelEncoder()


# In[26]:


df.AirportFrom=lc.fit_transform(df.AirportFrom)
df.AirportTo=lc.fit_transform(df.AirportTo)


# In[27]:


df.columns


# In[28]:


df.head(5)


# In[29]:


print(len(df),df.shape)


# In[30]:


plt.subplots(figsize=(20,15))
sns.heatmap(df.corr(),annot=True)


# In[31]:


y= df.Delay


# In[32]:


y.head()


# In[33]:


X= df.drop('Delay',axis=1)


# In[34]:


X.columns


# In[35]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.25,random_state=316)


# In[36]:


y_train


# In[37]:


y_train.shape


# In[38]:


from sklearn.linear_model import LogisticRegression
lg= LogisticRegression(solver='liblinear',multi_class='auto')


# In[39]:


lg.fit(X_train,y_train)
lg.score(X_train,y_train)


# In[40]:


lg.score(X_test,y_test)


# In[41]:


from sklearn.metrics import confusion_matrix


# In[43]:


y_predLG= lg.predict(X_test)
CM_LG= confusion_matrix(y_test,y_predLG)
sns.heatmap(CM_LG,center=True)
plt.show()
print('Confusion Matrix is\n', CM_LG)


# In[44]:


from sklearn.tree import DecisionTreeClassifier
dtc= DecisionTreeClassifier(criterion='entropy',max_depth=16,random_state=40)


# In[45]:


dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)


# In[46]:


dtc.score(X_test,y_test)


# In[47]:


y_predDT= dtc.predict(X_test)
CM_DT= confusion_matrix(y_test,y_predLG)
sns.heatmap(CM_DT,center=True)
plt.show()
print('Confusion Matrix is\n', CM_DT)


# In[48]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier(criterion = 'entropy', max_depth=18, n_estimators=400, random_state=44)


# In[49]:


rf.fit(X_train,y_train)
rf.score(X_train,y_train)


# In[50]:


rf.score(X_test,y_test)


# In[51]:


y_predRF= rf.predict(X_test)
CM_RF= confusion_matrix(y_test,y_predRF)
sns.heatmap(CM_RF,center=True)
plt.show()
print('Confusion Matrix is\n', CM_RF)


# In[52]:


from sklearn.ensemble import GradientBoostingClassifier
gb= GradientBoostingClassifier(n_estimators=300, max_depth=8, learning_rate=0.25, random_state=30)


# In[53]:


gb.fit(X_train,y_train)
gb.score(X_train,y_train)


# In[54]:


gb.score(X_test,y_test)


# In[55]:


y_predGB= gb.predict(X_test)
CM_GB= confusion_matrix(y_test,y_predGB)
sns.heatmap(CM_GB,center=True)
plt.show()
print('Confusion Matrix is\n', CM_GB)


# In[ ]:





# In[56]:


from sklearn.ensemble import VotingClassifier
vc= VotingClassifier(estimators=[('lg',lg),('dtc',dtc),('rf',rf),('gb',gb)],voting='hard')


# In[57]:


vc.fit(X_train,y_train)
vc.score(X_train,y_train)


# In[58]:


vc.score(X_test,y_test)


# In[ ]:





# VotingClassifierModel = VotingClassifier(estimators=[('GBCModel',GBCModel),('RFCModel',RandomForestClassifierModel),
#                                                      ('TDCModel',DecisionTreeClassifierModel)],
#                                          voting='hard')

# In[ ]:


##HyperParameterTuning models for better accuracy


# In[ ]:


##RandomSearchCV


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
n_estimators= [int(x) for x in np.linspace(start=200,stop=2000,num=10)]
max_features= ['auto','log2']
max_depth=[int(x) for x in np.linspace(10, 1000,10)]
min_samples_split= [1,2,5,10,14]
min_samples_leaf= [1,2,4,6,8]
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              'criterion':['entropy','gini']}
print(random_grid)


# In[ ]:


rf_randomcv= RandomizedSearchCV(estimator=rf,param_distributions=random_grid,n_iter=10,cv=3,verbose=2,random_state=100,n_jobs=-1)
rf_randomcv.fit(X_train,y_train)


# In[ ]:


best_random_grid=rf_randomcv.best_estimator_


# In[ ]:


from sklearn.metrics import accuracy_score
y_pred=best_random_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))


# In[ ]:





# In[ ]:


##GridSearchCV


# In[ ]:


from sklearn.model_selection import GridSearchCV

param_grid = {
    'criterion': [rf_randomcv.best_params_['criterion']],
    'max_depth': [rf_randomcv.best_params_['max_depth']],
    'max_features': [rf_randomcv.best_params_['max_features']],
    'min_samples_leaf': [rf_randomcv.best_params_['min_samples_leaf'], 
                         rf_randomcv.best_params_['min_samples_leaf']+2, 
                         rf_randomcv.best_params_['min_samples_leaf'] + 4],
    'min_samples_split': [rf_randomcv.best_params_['min_samples_split'] - 2,
                          rf_randomcv.best_params_['min_samples_split'] - 1,
                          rf_randomcv.best_params_['min_samples_split'], 
                          rf_randomcv.best_params_['min_samples_split'] +1,
                          rf_randomcv.best_params_['min_samples_split'] + 2],
    'n_estimators': [rf_randomcv.best_params_['n_estimators'] - 200, rf_randomcv.best_params_['n_estimators'] - 100, 
                     rf_randomcv.best_params_['n_estimators'], 
                     rf_randomcv.best_params_['n_estimators'] + 100, rf_randomcv.best_params_['n_estimators'] + 200]
}

print(param_grid)


# In[ ]:


rf=RandomForestClassifier()
grid_search=GridSearchCV(estimator=rf,param_grid=param_grid,cv=10,n_jobs=-1,verbose=2)
grid_search.fit(X_train,y_train)


# In[ ]:


grid_search.best_estimator_


# In[ ]:


best_grid=grid_search.best_estimator_


# In[ ]:


best_grid


# In[ ]:


y_pred=best_grid.predict(X_test)
print(confusion_matrix(y_test,y_pred))
print("Accuracy Score {}".format(accuracy_score(y_test,y_pred)))
print("Classification report: {}".format(classification_report(y_test,y_pred)))


# In[ ]:





# In[ ]:


#Multilayer Neural Network


# In[59]:


from sklearn.neural_network import MLPClassifier
MLP= MLPClassifier(hidden_layer_sizes=(10,10,10),max_iter=100)


# In[60]:


MLP.fit(X_train,y_train)


# In[61]:


MLP.score(X_train,y_train)


# In[62]:


MLP.score(X_test,y_test)


# In[ ]:




