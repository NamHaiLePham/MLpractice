
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from datetime import datetime
import plotly.express as px
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
# load the data
data=pd.read_csv('C:/Users/Hai/Desktop/USN/Semester 4/ML competition/data/sensor1.csv')

# missing value percentage
a=round((data.isnull().sum() * 100/ len(data)),2).sort_values(ascending=False)
# Remove relavant variables
df = data.drop(['sensor_15','sensor_50','sensor_51','Unnamed: 0'], axis = 1)
df.head()
Sensors = df.iloc[:,1:50]
sensorNames=Sensors.columns
# Replace missing values and normalize the dataset
names = df.iloc[:,1:50].columns
x = df.iloc[:,1:50].fillna(method = 'ffill')
scaler =StandardScaler()
pca=PCA()
pipeline=make_pipeline(scaler,pca)
pipeline.fit(x)
features = range(pca.n_components_)
_ = plt.figure(figsize=(15, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Importance of the Principal Components based on inertia")
plt.show()

# Calculate PCA with n components
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
#I already know that there are 3 classes of "NORMAL" vs "NOT NORMAL" which are combination of BROKEN" and"RECOVERING"
model =  IsolationForest(contamination=0.1)
model.fit(principalDf.values) 
principalDf['anomaly2'] = pd.Series(model.predict(principalDf.values))
# visualization
df['anomaly2'] = pd.Series(principalDf['anomaly2'].values, index=df.index)
a = df.loc[df['anomaly2'] == -1] #anomaly
print(a)
