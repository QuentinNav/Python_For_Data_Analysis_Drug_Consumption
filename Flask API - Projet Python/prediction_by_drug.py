from flask import Flask, render_template, request
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split,cross_val_score
from mapping_data import *


def ConvertFloatFeatures(name, features, df):
    features[df.columns.get_loc(name)]=float(features[df.columns.get_loc(name)])
    return features

def ConvertIntFeatures(name, features,df):
    features[df.columns.get_loc(name)]=int(features[df.columns.get_loc(name)])
    return features

def ComputeAccuracy(X,y,drug):

    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.2, random_state=0)
    model= LinearSVC()
    return round(cross_val_score(model, X_train, y_train, cv=5,n_jobs=-1,scoring='accuracy').mean(),4)

def modelPrediction(X,y, drug,test_features):

    model=LinearSVC().fit(X,y)

    return model.predict([test_features])

def dfCreation (df, features, drug, InitdrugList) :

    #creates a list with only the other drugs
    drugList=InitdrugList.copy()
    drugList.remove(drug)

    dfmodel=df.copy()

    #Drops the columns of the drugs we don't want
    dfmodel.drop(columns=drugList,inplace=True)

    #Drops the columns of the dataframe for which we don't have the value in the features list
    toDrop=[]
    for i in range(len(features)):
        if features[i]=="na":
            toDrop.append(dfmodel.columns[i])
    dfmodel.drop(columns=toDrop, inplace=True)
    features = list(filter(lambda x: x !=  "na", features))


    #Convert the values of features in integers of floats
    for i in ["age","gender","education","country","nscore","escore","oscore","ascore","cscore"]:
        if i in dfmodel.columns :
            features=ConvertIntFeatures(i,features,dfmodel)

    for i in ["impulsive","ss"]:
        if i in dfmodel.columns :
            features=ConvertFloatFeatures(i,features,dfmodel)

    #Add the features to the of the dataframe
    features.append(42)
    dfmodel.loc[1889]=features

    dfmodel[drug]=pd.to_numeric(dfmodel[drug],downcast="integer").astype('category')

    #Get_dummies :
    if "gender" in dfmodel.columns :
        dfmodel=pd.get_dummies(dfmodel,columns=["gender"])
        dfmodel=dfmodel.drop('gender_0', axis=1)
    if "country" in dfmodel.columns :
        dfmodel=pd.get_dummies(dfmodel, columns=["country"])
    if "ethnicity" in dfmodel.columns:
        dfmodel=pd.get_dummies(dfmodel, columns=["ethnicity"])

    return dfmodel


def ValidFeatures(features):
    for i in range(len(features)):

        if len(features[i])==0:
            return False
        if len(features)!=12:
            return False
    return True
