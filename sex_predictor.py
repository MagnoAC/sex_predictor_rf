#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all libraries that will be used

from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from scipy import stats
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import pickle


# In[2]:


#Function Used to encoding [1]
def encode(target):
    return unique.tolist().index(target)


# Data Import and Cleaning

# In[48]:


#import data
def main(input_file):
    missing_values = ["n/a","na","--", "NaN"] #list of possibles missing values
    data_raw = pd.read_csv(filename, sep =',', na_values = missing_values)
    data_fs = data_raw.copy()
    
    #removing not used features
    data_fs = data_fs.drop(['index'], axis =1).copy() #droping index 
    data_fs = data_fs.drop(['cp'], axis =1).copy() #droping cp
    
    #storing median values
    median_s = data_fs['slope'].median()
    median_c = data_fs['chol'].median()
    median_a = data_fs['age'].median()
    median_tr = data_fs['trestbps'].median()
    median_th= data_fs['thalach'].median()
    median_o = data_fs['oldpeak'].median()
    median_trf = data_fs['trf'].median()
    
    #filling NA values with the median
    data_fs['slope'].fillna(median_s, inplace = True) 
    data_fs['chol'].fillna(median_c, inplace = True) 
    data_fs['age'].fillna(median_a, inplace = True) #filling NA with the median
    data_fs['trestbps'].fillna(median_tr, inplace = True) #filling NA with the median
    data_fs['thalach'].fillna(median_th, inplace = True) #filling NA with the median
    data_fs['oldpeak'].fillna(median_o, inplace = True) #filling NA with the median
    data_fs['oldpeak'].fillna(median_trf, inplace = True) #filling NA with the median
    
    #Removing NA values for discrete features
    data_fs = data_fs[(data_fs['fbs'].notna())    |
                     (data_fs['restecg'].notna()) |
                     (data_fs['exang'].notna())   |
                     (data_fs['ca'].notna())      |
                     (data_fs['thal'].notna())    |
                     (data_fs['nar'].notna())     |
                     (data_fs['hc'].notna())      |
                     (data_fs['sk'].notna())       ]

    #Encoding
    enco_data = data_fs.copy()
    unique = enco_data['sex'].unique()
    enco_data['sex'] = enco_data.sex.str.upper()
    enco_data['sex'] = enco_data['sex'].apply(encode)
    
    #Feature selection
    eda = enco_data.copy()
    eda = eda[['age','sex','trestbps','chol','exang','oldpeak','slope','thalach', 'trf']]
    
    #Outlier Removal
    df = eda.copy()
    df = df[(np.abs(stats.zscore(df[['age'] + ['trestbps'] +['chol'] + ['oldpeak'] + ['thalach']] + ['trf']) < 2.2).all(axis=1))] #outlier removal
    
    #Separation of Target and Features
    X = df.iloc[:, df.columns != 'sex']  #independent columns
    y = df.iloc[:,1]    #target column i.e sex
    
    #Prediction CSV Output
    loaded_model = pickle.load(open("rf_model.pkl", 'rb'))
    y_pred = loaded_model.predict(X)
    result = df.copy()
    result[['sex']] = y_pred
    result = result[['sex']]
    result[['sex']] = result['sex'].replace({0: 'F', 1: 'M'})
    result('newsample.csv', index = False)

