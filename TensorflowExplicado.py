# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 09:24:17 2023

@author: noriy
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import OneHotEncoder 

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

#labels will be used as an argument in the Tensorflow function, it will indicate the amount of variables(inputs) we'll be using
labels=list(x_test.columns.values)


#Defining the model
model=keras.Sequential([keras.layers.Dense(1,input_shape=(len(labels),),
                                           activation='sigmoid')])
model.compile(optimizer='adam',loss='binary_crossentropy')


#Fitting the model
hist=model.fit(x_train,y_train,epochs=1000,validation_split=0.2)



#Plotting the loss function
def plot_loss(model):
    plt.plot(model.history['loss'],label="Loss")
    plt.plot(model.history['val_loss'], label='val_loss')
    plt.ylim([0,1])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.grid(True)
    
plot_loss(hist)
    
    
          
print(model.summary())
predictions=model.predict(x_test)


#Defining the threshold
for i in range(len(predictions)):
    predictions[i]=1 if predictions[i]>=0.75 else 0
predictions = [int(x[0]) for x in predictions]
print(predictions)

#Accuracy check
def accuracy(y_pred,y_test):
    return np.sum((y_pred==y_test)/len(y_test))

accuracy(predictions, y_test)



