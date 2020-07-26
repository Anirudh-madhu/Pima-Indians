#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
from sklearn.preprocessing import StandardScaler


# In[7]:


names=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Outcome']


# In[8]:


data= pd.read_csv("pima-indians-diabetes.csv", names=names)


# In[9]:


data


# In[14]:


## average pregancy
data['Pregnancies'].mean()


# In[17]:


data.isnull().sum() ## Missing values


# In[18]:


##Average blood pressure found in pima indians
data['BloodPressure'].mean()


# In[25]:


data['Outcome'].hist(grid=False)


# In[28]:


data.describe()


# In[30]:


data['Age'].unique()


# In[40]:


data[data.BloodPressure>=90].head()  ##Normal Blood pressure 


# In[41]:


data.SkinThickness.mean()   ##Triceps skin fold thickness (mm)


# In[42]:


data[data.Insulin<=25]    # Fasting Insulin range


# In[46]:


X= data.iloc[:,0:8]
y= data.iloc[:,8]


# In[51]:


a=StandardScaler()
standard=a.fit_transform(X)                  #Standardizing the data


# In[53]:


from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# In[61]:


model=Sequential()
model.add(Dense(8,input_dim=8, activation='relu'))
model.add(Dense(4, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

adam=Adam(lr=0.01)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])


# In[63]:


model.fit(standard,y, epochs=100,batch_size=32)


# In[65]:


import pickle



# In[70]:


from sklearn.linear_model import LogisticRegression


# In[71]:


lg= LogisticRegression()


# In[72]:


lg.fit(standard,y)


# In[73]:




# In[76]:


from sklearn.model_selection import train_test_split


# In[78]:


X_train, X_test, y_train, y_test= train_test_split(standard,y,test_size=0.3, random_state=145)


# In[80]:


lg.fit(X_train, y_train)


# In[81]:


pred=lg.predict(X_test)


# In[83]:


ourfile= open("dogs.pkl", "wb")
pickle.dump(lg, ourfile)
ourfile.close()


# In[84]:


infile = open("dogs.pkl",'rb')
new_dict = pickle.load(infile)
infile.close()


# In[85]:


print(new_dict)


# In[ ]:




