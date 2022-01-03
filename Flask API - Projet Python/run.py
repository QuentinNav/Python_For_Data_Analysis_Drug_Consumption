# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 14:57:24 2021

@author: bruno
"""
from flask import Flask, render_template, request
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split,cross_val_score
from mapping_data import *
from prediction_by_drug import *


app = Flask(__name__)

features=[]




@app.route('/')
def home():
    return render_template("home.html")


@app.route('/Data_analysis')
def data_analysis():
    return render_template("data_analysis.html")


@app.route('/drugged_estimator')
def form():
        return render_template('form.html')
    
@app.route('/ppt_report')
def ppt():
    return render_template('ppt_report.html')


@app.route('/result', methods=['POST'])
def result():
    final="incorrect completion of the questionnaire"
    if request.method == 'POST':
        
        features=[]
        
        features.append(request.form['age'])
        features.append(request.form['gender'])
        features.append(request.form['education'])
        features.append(request.form['country'])
        features.append(request.form['ethnicity'])        
        features.append(request.form['sensation'])
        features.append(request.form['impulsiveness']) 
        features.append(request.form['nscore'])
        features.append(request.form['escore'])
        features.append(request.form['oscore'])
        features.append(request.form['ascore'])
        features.append(request.form['cscore'])
        
        
        features.insert(11, features.pop(6))
        features.insert(11, features.pop(5))
        
        final =IncompleteFeaturesIA(features)
            
    return render_template('result.html',
                            drug1=final["amphet"],
                            drug2=final["amyl"],
                            drug3=final["benzos"],
                            drug4=final["cannabis"],
                            drug5=final["coke"],
                            drug6=final["crack"],
                            drug7=final["ecstasy"],
                            drug8=final["heroin"],
                            drug9=final["ketamine"],
                            drug10=final["lsd"],
                            drug11=final["meth"],
                            drug12=final["vsa"]
                            )


def toString(tab) :
    results={}
    
    
    
    for i in tab :

        if i[2]==0:
            i[2]="Non-user"
        else:
            i[2]="User"
        i[1]=round(i[1]*100,2)
        
        results[str(i[0])]= str(i[0])+" : "+str(i[2])+" (with "+str(i[1])+"% accuracy)"
        
    print(results)
    return results
        




def IncompleteFeaturesIA(features):
    df=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/00373/drug_consumption.data")

    df.loc[len(df)]=df.columns
    df.columns=(["id","age","gender","education","country","ethnicity","nscore","escore","oscore","ascore","cscore","impulsive","ss",
     "alcohol","amphet","amyl","benzos","caff","cannabis","choc","coke","crack","ecstasy","heroin","ketamine","legalh",
     "lsd","meth","mushrooms","nicotine","semer","vsa"])
    df["id"]=df["id"].astype(int)
    df = df.set_index("id")
    df.sort_index(inplace=True)
    for i in range(12):
        df.iloc[:,i]=df.iloc[:,i].astype(float)
    
    df["age"]=df.age.map(age)
    df["gender"]=df.gender.map(gender)
    df["education"]=df.education.map(education)
    df["country"]=df.country.map(country)
    df["ethnicity"]=df.ethnicity.map(ethnicity)
    df["nscore"]=df.nscore.map(nscore)
    df["escore"]=df.escore.map(escore)
    df["oscore"]=df.oscore.map(oscore)
    df["ascore"]=df.ascore.map(ascore)
    df["cscore"]=df.cscore.map(cscore)
    
    drug_list=["alcohol","amphet","amyl","benzos","caff","cannabis","choc","coke","crack","ecstasy","heroin","ketamine","legalh",
              "lsd","meth","mushrooms","nicotine","semer","vsa"]
    
    for i in drug_list:
        df[i].replace(drogue, inplace=True)
      
    dfnan = df[df.isna().any(axis=1)]
    #we observe that Nan value wich correspond to unreferenced value in the documentation correspond only to the ID 1.
    #Thus we have decided to drop this row from our dataframe.
    df=df.iloc[1:]
    
    
    dfml=df.copy()
    dfml.drop(columns=['caff', 'choc','legalh','mushrooms','nicotine','alcohol',],inplace=True)
    dfml["semer"]=dfml["semer"].apply(lambda x: 1 if x in ('Used in Last Month', 'Used over a Decade Ago','Used in Last Year', 'Used in Last Decade', 'Used in Last Day','Used in Last Week') else 0)
    dfml.drop(dfml[dfml['semer']==1].index, inplace=True)
    #8 liers have been removed
    dfml.drop(columns=['semer'],inplace=True)
    drugList=['amphet','amyl', 'benzos', 'cannabis', 'coke', 'crack', 'ecstasy', 'heroin','ketamine', 'lsd', 'meth', 'vsa']
    for i in drugList:
        dfml[i]=dfml[i].map(User)
    
    for i in drugList : 
        dfml.loc[dfml[i]!="User",i]=0
        dfml.loc[dfml[i]=="User",i]=1
    
        
    #Then we replace other cateorical variables by continuous value.
    
    dfml["age"]=dfml["age"].apply(lambda x: random.randrange(25, 34) if x=='25-34' else x)
    dfml["age"]=dfml["age"].apply(lambda x: random.randrange(18, 24) if x=='18-24' else x)
    dfml["age"]=dfml["age"].apply(lambda x: random.randrange(35, 44) if x=='35-44' else x)
    dfml["age"]=dfml["age"].apply(lambda x: random.randrange(45, 54) if x=='45-54' else x)
    dfml["age"]=dfml["age"].apply(lambda x: random.randrange(55, 64) if x=='55-64' else x)
    dfml["age"]=dfml["age"].apply(lambda x: random.randrange(65, 100) if x=='65+' else x)
    
    edu_dict = {
        "Left school before 16 years": 0,
        "Left school at 16 years":1,
        'Left school at 17 years':2,
        "Left school at 18 years":3,
        "Some college or university, no certificate or degree":4,
        "Professional certificate/ diploma":5,
        "University degree":6,
        "Masters degree":7,
        "Doctorate degree":8
        }
    
    dfml["gender"] = dfml["gender"].astype('category')
    dfml["gender"] = dfml["gender"].cat.codes
    
    dfml["country"] = dfml["country"].astype('category')
    dfml["country"] = dfml["country"].cat.codes
    
    for i in drugList : 
        dfml[i]=pd.to_numeric(dfml[i],downcast="integer").astype("category")
        
    
    dfml["education"].replace(edu_dict, inplace=True)
    
    results=[]
    
    for i in drugList : 
        resultDrug=[]
        resultDrug.append(i)
        
        dfmodel=dfCreation(dfml,features,i,drugList)
        
        X=dfmodel.drop(i,axis=1)
        X=pd.DataFrame(MinMaxScaler().fit_transform(X),columns=X.columns)
        test_features =X.iloc[-1].tolist()
        X.drop(X.tail(1).index, inplace=True)
        
        y=dfmodel[i]
        y.drop(y.tail(1).index, inplace=True)
        
        resultDrug.append(ComputeAccuracy(X, y, i))
        
        resultDrug.append(modelPrediction(X,y,i, test_features)[0])
        
        results.append(resultDrug)
        
    
    return toString(results)





    



if __name__ == '__main__':
    app.run(host='localhost', port=5000)