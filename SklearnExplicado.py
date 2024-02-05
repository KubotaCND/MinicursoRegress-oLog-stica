# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 22:08:44 2023

@author: noriy
"""
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #para plotar gráficos tipo hist
import seaborn as sns #para plotar gráficos mais simplificado de usar, heat
import matplotlib as plot #para plotar gráficos, tipo simples hist
import sklearn

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

#Spliting x and y
x=main.iloc[:,[0,2,3,4,5,6,7,8]]
y=main.iloc[:,1]



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#With this library we have a function that splits our data set to us
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

#Defining our model
Previs_titanic=LogisticRegression()
Previs_titanic.fit(x_train,y_train)


testing=Previs_titanic.predict(x_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, testing))
from sklearn.metrics import classification_report
print(classification_report(y_test, testing))

