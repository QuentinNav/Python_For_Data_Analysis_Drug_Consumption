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


app = Flask(__name__)

features=[]


@app.route('/')
def form():
        return render_template('form.html')


@app.route('/result', methods=['POST'])
def result():
    final="incorrect completion of the questionnaire"
    if request.method == 'POST':
        
        features=[]
        features.extend(request.form.getlist('mycheckbox'))
        features.append(request.form['nscore'])
        features.append(request.form['escore'])
        features.append(request.form['oscore'])
        features.append(request.form['ascore'])
        features.append(request.form['cscore'])
        
        if ValidFeatures(features):
            features.insert(11, features.pop(6))
            features.insert(11, features.pop(5))
            final =IA(features)
            
    return render_template('result.html',drug=final)



def ValidFeatures(features):
    for i in range(len(features)):
        
        if len(features[i])==0:
            return False
        if len(features)!=12:
            return False
    return True


def IA(features):
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
    df=df.iloc[1:]
    
    dfml=df.copy()
    dfml.drop(columns=['caff', 'choc','legalh','mushrooms','nicotine','alcohol',],inplace=True)
    dfml["semer"]=dfml["semer"].apply(lambda x: 1 if x in ('Used in Last Month', 'Used over a Decade Ago','Used in Last Year', 'Used in Last Decade', 'Used in Last Day','Used in Last Week') else 0)
    dfml.drop(dfml[dfml['semer']==1].index, inplace=True)
    dfml.drop(columns=['semer'],inplace=True)
    drugList=['amphet','amyl', 'benzos', 'cannabis', 'coke', 'crack', 'ecstasy', 'heroin','ketamine', 'lsd', 'meth', 'vsa']
    for i in drugList:
        dfml[i]=dfml[i].map(User)

    condition = ((dfml['amphet'] != 'User') & 
             (dfml['amyl'] != 'User') &
             (dfml['benzos'] != 'User') &
             (dfml['cannabis'] != 'User') &
             (dfml['coke'] != 'User') &
             (dfml['crack'] != 'User') &
             (dfml['ecstasy'] != 'User') &
             (dfml['heroin'] != 'User') &
             (dfml['ketamine'] != 'User') &
             (dfml['lsd'] != 'User') &
             (dfml['meth'] != 'User') &
             (dfml['vsa'] != 'User'))


    dfml.loc[condition, 'Drug'] = 0
    dfml.loc[~condition, 'Drug'] = 1

    for i in drugList:
        dfml.drop(columns=[i],inplace=True)


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
    for i in range(len(features)):
        if i!=4:
            if (i!=11) & (i!=10):
                features[i]=int(features[i])
            else:
                features[i]=float(features[i])
    features.append(42)


    dfml["gender"] = dfml["gender"].astype('category')
    dfml["gender"] = dfml["gender"].cat.codes

    dfml["country"] = dfml["country"].astype('category')
    dfml["country"] = dfml["country"].cat.codes

    dfml['Drug']=pd.to_numeric(dfml['Drug'],downcast='integer').astype('category')
    dfml["education"].replace(edu_dict, inplace=True)

    dfml.loc[1889]=features


    dfml=pd.get_dummies(dfml,columns=['gender', 'country',"ethnicity"])
    dfml=dfml.drop('gender_0', axis=1)

    X=dfml.drop('Drug', axis=1)
    X = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)
    
    

    test_features=X.iloc[-1].tolist()
    X.drop(X.tail(1).index,inplace=True)
    y=dfml["Drug"]

    y.drop(y.tail(1).index,inplace=True)
    model= LinearSVC().fit(X,y)
    
    drugged=model.predict([test_features])
    
   
    if drugged==[1]:
        return 'The person is probably considered as drugged (82% accuracy)'
    else:
        return 'The person is probably considered as clean (82% accuracy)'







    



age= {
-0.95197 : "18-24",
-0.07854 :"25-34",
0.49788 :"35-44",
1.09449 :"45-54",
1.82213 :"55-64",
2.59171 :"65+",
}

gender= {
0.48246 :"Female" ,
-0.48246: "Male",
}

education= {
-2.43591 :"Left school before 16 years" ,
-1.73790 :"Left school at 16 years" ,
-1.43719 :"Left school at 17 years" ,
-1.22751 :"Left school at 18 years" ,
-0.61113: "Some college or university, no certificate or degree" ,
-0.05921 :"Professional certificate/ diploma",
0.45468 :"University degree" ,
1.16365 :"Masters degree",
1.98437 :"Doctorate degree" ,
}

country= {
-0.09765: "Australia",
0.24923: "Canada" ,
-0.46841 :"New Zealand" ,
-0.28519: "Other" ,
0.21128 :"Republic of Ireland" ,
0.96082 :"UK" ,
-0.57009 :"USA",
}

ethnicity= {
-0.50212 :"Asian",
-1.10702: "Black" ,
1.90725: "Mixed-Black/Asian" ,
0.12600 :"Mixed-White/Asian" ,
-0.22166: "Mixed-White/Black" ,
0.11440: "Other",
-0.31685 :"White" ,
}

nscore= {
-3.46436 :12,
-0.67825 :29,
1.02119:46,
-3.15735 :13,
-0.58016 :30,
1.13281:47,
-2.75696 :14,
-0.46725 :31,
 1.23461:48,
 -2.52197 :15,
 -0.34799 :32,
 1.37297:49,
 -2.42317 :16,
 -0.24649 :33,
 1.49158:50,
 -2.34360 :17,
 -0.14882 :34,
 1.60383:51,
 -2.21844 :18,
 -0.05188 :35,
 1.72012:52,
 -2.05048 :19,
 0.04257 :36,
 1.83990:53,
 -1.86962 :20,
 0.13606 :37,
 1.98437:54,
 -1.69163 :21,
 0.22393 :38,
 2.12700:55,
 -1.55078 :22,
 0.31287 :39,
 2.28554:56,
 -1.43907 :23,
 0.41667 :40,
 2.46262:57,
 -1.32828:24 ,
 0.52135 :41,
 -1.19430:25 ,
2.61139:58,
 0.62967 :42,
 2.82196:59,
 -1.05308:26 ,
 0.73545 :43,
 3.27393:60,
 -0.92104:27 ,
 0.82562:44,
 -0.79151 :28,
 0.91093:45,
}

escore= {
-3.27393:16,
-3.00537:18,
-2.72827:19,
-2.5383:20,
-2.44904:21,
-2.32338:22,
-2.21069:23,
-2.11437:24,
-2.03972:25,
-1.92173:26,
-1.7625:27,
-1.6334:28,
-1.50796:29,
-1.37639:30,
-1.23177:31,
-1.09207:32,
-0.94779:33,
-0.80615:34,
-0.69509:35,
-0.57545:36,
-0.43999:37,
-0.30033:38,
-0.15487:39,
0.00332:40,
0.16767:41,
0.32197:42,
0.47617:43,
0.63779:44,
0.80523:45,
0.96248:46,
1.11406:47,
1.2861:48,
1.45421:49,
1.58487:50,
1.74091:51,
1.93886:52,
2.127:53,
2.32338:54,
2.57309:55,
2.8595:56,
3.00537:58,
3.27393:59
}

ascore ={
-3.46436	:	12,
-3.15735	:	16,
-3.00537	:	18,
-2.90161	:	23,
-2.78793	:	24,
-2.70172	:	25,
-2.53830	:	26,
-2.35413	:	27,
-2.21844	:	28,
-2.07848	:	29,
-1.92595	:	30,
-1.77200	:	31,
-1.62090	:	32,
-1.47955	:	33,
-1.34289	:	34,
-1.21213	:	35,
-1.07533	:	36,
-0.91699	:	37,
-0.76096	:	38,
-0.60633	:	39,
-0.45321	:	40,
-0.30172	:	41,
-0.15487	:	42,
-0.01729	:	43,
0.13136	:	44,
0.28783	:	45,
0.43852	:	46,
0.59042	:	47,
0.76096	:	48,
0.94156	:	49,
1.11406	:	50,
1.2861	:	51,
1.45039	:	52,
1.61108	:	53,
1.81866	:	54,
2.03972	:	55,
2.23427	:	56,
2.46262	:	57,
2.75696	:	58,
3.15735	:	59,
3.46436	:	60,
}

oscore= {
 -3.27393 :24,
 -1.11902 :37,
 0.58331:50,
 -2.85950 :26,
 -0.97631:39,
 0.72330:51,
 -2.63199 :28,
 -0.84732 :40,
 0.88309:52,
 -2.39883: 29,
 -0.71727 :41,
 1.06238:53,
 -2.21069 :30,
 -0.58331 :42,
 1.24033:54,
 -2.09015 :31,
 -0.45174 :43,
 1.43533:55,
 -1.97495: 32,
 -0.31776: 44,
 1.65653:56,
 -1.82919: 33,
 -0.17779: 45,
 1.88511:57,
 -1.68062: 34,
 -0.01928:46,
 2.15324:58,
 -1.55521:35,
 0.14143: 47,
 2.44904:59,
 -1.42424 :36,
 0.29338 :48,
 2.90161:60,
 -1.27553: 37,
 0.44585:49,
}

cscore = {
-3.46436	:	17,
-3.15735	:	19,
-2.90161	:	20,
-2.72827	:	21,
-2.57309	:	22,
-2.42317	:	23,
-2.30408	:	24,
-2.18109	:	25,
-2.04506	:	26,
-1.92173	:	27,
-1.78169	:	28,
-1.64101	:	29,
-1.51840	:	30,
-1.38502	:	31,
-1.25773	:	32,
-1.13788	:	33,
-1.01450	:	34,
-0.89891	:	35,
-0.78155	:	36,
-0.65253	:	37,
-0.52745	:	38,
-0.40581	:	39,
-0.27607	:	40,
-0.14277	:	41,
-0.00665	:	42,
0.12331	:	43,
0.25953	:	44,
0.41594	:	45,
0.58489	:	46,
0.7583	:	47,
0.93949	:	48,
1.13407	:	49,
1.30612	:	50,
1.46191	:	51,
1.63088	:	52,
1.81175	:	53,
2.04506	:	54,
2.33337	:	55,
2.63199	:	56,
3.00537	:	57,
3.46436	:	59,
}

drogue = {
"CL0": "Never Used" ,
"CL1" :"Used over a Decade Ago" ,
"CL2" :"Used in Last Decade",
"CL3" :"Used in Last Year" ,
"CL4": "Used in Last Month" ,
"CL5" :"Used in Last Week" ,
"CL6" :"Used in Last Day" ,
}

User = {
"Never Used": "Non-user" ,
"Used over a Decade Ago":"Non-user" ,
"Used in Last Decade":"User",
"Used in Last Year":"User" ,
"Used in Last Month": "User" ,
"Used in Last Week":"User" ,
"Used in Last Day" :"User" ,
}



if __name__ == '__main__':
    app.run(host='localhost', port=5000)