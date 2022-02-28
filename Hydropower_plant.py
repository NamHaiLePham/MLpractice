# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 08:56:14 2022

@author: Hai
"""


import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
import statistics
url='C:/Users/Hai/Desktop/USN/Semester 4/ML competition/data/Study cases.xlsx'
DF=pd.read_excel(url)
data=DF.values

#Check opearation condition (Y)
#Check sensor realiability
#Steps: check sensor realiability get the anormaly values including wrong sensors and not running gen
#First check missing values

#Check misisng values
# missing value percentage
a=round((DF.isnull().sum() * 100/ len(DF)),2).sort_values(ascending=False)
print(a)#See the missing values in column and annouce that these sensor have problem
# Remove relavant variables
df = DF.drop(['Rotating speed (rph)','Power (MW)','Temperature (degree)'], axis = 1)
Sensors = df.iloc[:,1:11]
sensorNames=Sensors.columns
# Replace missing values and normalize the dataset
names = df.iloc[:,1:11].columns
x = df.iloc[:,1:11].fillna(method = 'ffill')


#Check sensor realiability of x which is the data removed missing values
clf = IsolationForest(n_estimators=200,contamination=0.1,max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
clf.fit(x)
score=clf.decision_function(x)
plt.hist(score, bins=100) #get bins components in data to draw score
df['scores'] = score #df is the data after dropping missing value
a=df.query('scores<0.1')#tuning the isolation forest
print(a)#a is the event has the anomaly values, includes the wrong sensors, gen not run

#if gen not run, the output is 0, remove the output of a
m=a[(a['Current (kA)'] <= 0) & (a['Gen Temperature (degree)'] <= 0)]#this is stop operation 
a.drop(m.index) #drop gen not run, the last is wrong sensor.








  
 
      

