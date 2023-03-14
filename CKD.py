# 1. librairies import

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression 
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split


def preprocessing(path,df_test):
    if df_test.empty == True :
        drop_class = 0
        dataset = pd.read_csv(path,na_values='?')
    else : 
        drop_class = 1
        dataset = df_test
    df_num_col = []
    df_cat_col = []
    for col in dataset.columns :
        if dataset[col].dtype == 'object':
            df_cat_col.append(col)
        else:
            df_num_col.append(col)

    data_full_remplace = dataset.copy()

    data_full_remplace = data_full_remplace.replace('yes',1)
    data_full_remplace = data_full_remplace.replace('no',0)

    data_full_remplace = data_full_remplace.replace('present',1)
    data_full_remplace = data_full_remplace.replace('notpresent',0)

    data_full_remplace = data_full_remplace.replace('normal',1)
    data_full_remplace = data_full_remplace.replace('abnormal',0)

    data_full_remplace = data_full_remplace.replace('good',1)
    data_full_remplace = data_full_remplace.replace('poor',0)

    data_full_remplace = data_full_remplace.replace('ckd',1)
    data_full_remplace = data_full_remplace.replace('notckd',0)

    for col in df_num_col :
        data_full_remplace[col]= data_full_remplace[col].fillna(data_full_remplace[col].mean())

    # 1ere idée pour les models : drop la colonne Red_Blood_Cells et Pus_Cell qui contient 38% et 18% d'inconnues
    drop_col = ['Red_Blood_Cells','Pus_Cell']
    # les autres on remplace par la médiane
    data_full_remplace_drop = data_full_remplace.copy()
    data_full_remplace_drop=data_full_remplace_drop.drop(['Red_Blood_Cells','Pus_Cell'],axis=1)

    for col in df_cat_col :
        if col != 'Red_Blood_Cells' and col != 'Pus_Cell' :
            data_full_remplace_drop[col]= data_full_remplace_drop[col].fillna(data_full_remplace_drop[col].median())
    missing_values = data_full_remplace_drop.isnull().sum()
    if 'Class' in data_full_remplace_drop.columns :
        if missing_values['Class'] != 0 or drop_class==1:
            data_full_remplace_drop = data_full_remplace_drop.drop(['Class'],axis = 1)
    
    return(data_full_remplace_drop)


def return_pred(dataset):
    dataset.to_csv('out.csv',index=False)

def train_data(dataset):
    X = dataset.copy().drop(['Class'],axis=1)
    y = dataset['Class']
    train_X, val_X, train_y, val_y = train_test_split(X,y,random_state=0,train_size=0.8, test_size=0.2)    
    lr = LogisticRegression(C= 78.47599703514607, max_iter= 2000, penalty='l2', solver= 'liblinear')
    knn = KNeighborsClassifier(algorithm= 'auto', n_neighbors= 3, p= 1, weights= 'uniform')
    svc = SVC(probability=True,C= 0.1, kernel= 'linear')
    rf = RandomForestClassifier(bootstrap= False, max_depth= 10, min_samples_leaf= 2, min_samples_split= 2, n_estimators= 400)
    xgb = XGBClassifier(colsample_bytree= 0.75, gamma= 0.5, learning_rate= 0.5, max_depth= None, min_child_weight= 0.01, n_estimators= 450, reg_alpha= 1, reg_lambda= 10, sampling_method= 'uniform', subsample= 0.65)
    best_lr = lr.fit(train_X,train_y)
    best_knn = knn.fit(train_X,train_y)
    best_svc = svc.fit(train_X,train_y)
    best_rf = rf.fit(train_X,train_y)
    best_xgb = xgb.fit(train_X,train_y)
    voting = VotingClassifier(estimators=[('knn',best_knn),('rf',best_rf),('svc',best_svc),('lr',best_lr),('xgb',best_xgb)],voting='hard')
    voting.fit(train_X,train_y)
    return(voting)

def prediction(path_train,path_test):
    print('Prediction on the file : ',path_test)
    data_test = preprocessing(path_test,pd.DataFrame({}))
    data_full_remplace = preprocessing(path_train,pd.DataFrame({}))
    voting = train_data(data_full_remplace)
    preds = voting.predict(data_test)
    data_test['Class']=preds
    return(data_test)

if __name__ == "__main__":
    path_train = 'dataset_full.csv'
    path_test ='dataset_test.csv'
    data_final = prediction(path_train,path_test)
    return_pred(data_final)
    for i in range (len(data_final)):
        if data_final.loc[i,'Class']==1:
            print('The patient {} has more chances to get CDK'.format(i))
        else : 
            print('The patient {} has less chances to get CKD'.format(i))


