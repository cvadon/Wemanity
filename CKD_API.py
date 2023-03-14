# 1. librairies import

import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt
from CKD_class import CKD_class_values
import uvicorn
from fastapi import FastAPI
# Model training


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn.naive_bayes import GaussianNB 
from sklearn.linear_model import LogisticRegression 
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split

# 2. create the app object

app = FastAPI()

# 3. index the route


@app.get('/')
def index():
    return{'message': ' Go to http://127.0.0.1:8000/docs '}



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
    data_test = preprocessing(path_test,pd.DataFrame({}))
    data_full_remplace = preprocessing(path_train,pd.DataFrame({}))
    voting = train_data(data_full_remplace)
    preds = voting.predict(data_test)
    data_test['Class']=preds
    return(preds)

def prediction_from_entries(df_test):

    path_train = 'dataset_full.csv'
    path_test='dataset_test.csv'
    data_test = preprocessing(path_test,df_test)
    data_full_remplace = preprocessing(path_train,pd.DataFrame({}))
    voting = train_data(data_full_remplace)
    preds = voting.predict(data_test)
    data_test['Class']=preds
    return(preds)

# 4. prediction

@app.post('/predict_for_one_patient')
def prediction1(data: CKD_class_values):
    file_train = "dataset_full.csv"
    df_train = pd.read_csv(file_train,na_values='?')
    columns_total = df_train.columns
    data = data.dict()
    Age = data['Age']
    Blood_Pressure = data['Blood_Pressure']
    Specific_Gravity = data['Specific_Gravity']
    Albumin = data['Albumin']
    Sugar = data['Sugar']
    Red_Blood_Cells = data['Red_Blood_Cells']
    Pus_Cell = data['Pus_Cell']
    Pus_Cell_clumps = data['Pus_Cell_clumps']
    Bacteria = data['Bacteria']
    Blood_Glucose_Random = data['Blood_Glucose_Random']
    Blood_Urea = data['Blood_Urea']
    Serum_Creatinine = data['Serum_Creatinine']
    Sodium = data['Sodium']
    Potassium = data['Potassium']
    Hemoglobin = data['Hemoglobin']
    Packed_Cell_Volume = data['Packed_Cell_Volume']
    White_Blood_Cell_Count = data['White_Blood_Cell_Count']
    Red_Blood_Cell_Count = data['Red_Blood_Cell_Count']
    Hypertension = data['Hypertension']
    Diabetes_Mellitus = data['Diabetes_Mellitus']
    Coronary_Artery_Disease = data['Coronary_Artery_Disease']
    Appetite = data['Appetite']
    Pedal_Edema = data['Pedal_Edema']
    Anemia = data['Anemia']
    Class = data['Class']
    df_to_test = pd.DataFrame([[Age, Blood_Pressure, Specific_Gravity,  Albumin ,  Sugar ,
        Red_Blood_Cells ,  Pus_Cell ,  Pus_Cell_clumps ,  Bacteria ,
        Blood_Glucose_Random ,  Blood_Urea ,  Serum_Creatinine ,  Sodium ,
        Potassium ,  Hemoglobin ,  Packed_Cell_Volume ,
        White_Blood_Cell_Count ,  Red_Blood_Cell_Count ,  Hypertension ,
        Diabetes_Mellitus ,  Coronary_Artery_Disease ,  Appetite ,
        Pedal_Edema ,  Anemia ,Class]], columns=columns_total)
    pred = prediction_from_entries(df_to_test)
    if pred[0] == 1 :
        prediction = 'Patient has more chances to get CKD'
    else:
        prediction = 'Patient has less chances to get CKD'
    return{'prediction': prediction}

@app.post('/predict_from_csv')
def prediction2(file_test="dataset_test.csv", file_train="dataset_full.csv"):
    pred = prediction(file_train,file_test)
    for i in range(len(pred)):
        if pred[i] == 1:
            prediction_total += [i,'Patient has more chances to get CKD']
    return{'message': f'les predictions sont: {prediction_total}'}


# 5. Run the API
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

# uvicorn CKD_API:app --reload
    

