# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:42:14 2023

@author: noriy
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 08:43:31 2023

@author: noriy
"""
import numpy as np

def sigmoid(x):
   return (1/(1+np.exp(-x)))

class LogisticRegression():
    
    def __init__(self,lr=0.001,n_iters=1000):
        self.lr= lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
        
    def fit(self, X, y):
        n_samples, n_features= X.shape
        self.weights=np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            linear_predictions=np.dot(X, self.weights)+self.bias
            predictions=sigmoid(linear_predictions)
            
            dw=(1/n_samples)*np.dot(X.T,(predictions-y))
            db=(1/n_samples)*np.sum(predictions-y)
            
            self.weights=self.weights-self.lr*dw
            self.bias=self.bias-self.lr*db
        
        
    def predict(self,X):
        linear_predictions=np.dot(X, self.weights)+self.bias
        y_pred=sigmoid(linear_predictions)
        class_pred=[0 if y<=0.5 else 1  for y in y_pred]
        return class_pred
    
    
def accuracy(y_pred,y_test):
    return np.sum((y_pred==y_test)/len(y_test))





#%%

from Logistic import LogisticRegression
import LogisticRegression
import pandas as pd
import numpy as np
#In this script the 'test.csv' part of the Titanic data from Kaggle wasn't used
#since this is a test to compare how different logistic regressions work and their accuracy

#Opening and reading the data
main=pd.read_csv('S:/Projeto Python Data Science/Titanic/titanic/train.csv')
#Cleaning the null data and changing the letters to numbers
main.isnull().sum()
main=main.dropna(subset=['Embarked'])
main['Age'].fillna(value=main['Age'].mean(),inplace=True)
main=main.drop(['Cabin','Ticket','Name'],axis=1)
main.isnull().sum()
main.head()
main.head().describe()
change={"male":0,"female":1}
main['Sex']=main['Sex'].map(change)
cg={"C":1,"Q":2,"S":3}
main['Embarked']=main['Embarked'].map(cg)
main.head()
889*0.80

#Dividing the training ser file into train and test subset
x_train=main.iloc[0:711,[0,2,3,4,5,6,7,8]]
y_train=main.iloc[0:711,1]


x_test=main.iloc[711:889,[0,2,3,4,5,6,7,8]]
y_test=main.iloc[711:889,1]


#lr=0.1  n_iters=700
regr=LogisticRegression(lr=0.05,n_iters=700)
regr.fit(x_train,y_train)
y_pred=regr.predict(x_test)


acc=accuracy(y_pred, y_test)
print(acc)
