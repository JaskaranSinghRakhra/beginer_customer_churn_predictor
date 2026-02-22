import os 
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np


if not os.path.exists("model.pkl"):
    ##make A model
    encode=OneHotEncoder()
    data=pd.read_csv("train.csv")
    cat_data=data[['InternetService',"PaymentMethod","Contract"]]
    encoded_data=encode.fit_transform(cat_data)
    data_encoded=pd.DataFrame(encoded_data.toarray(),columns=['DSL', 'Fiber optic', 'No','Bank transfer (automatic)', 'Credit card (automatic)',
            'Electronic check', 'Mailed check','Month-to-month', 'One year', 'Two year'],index=data.index)
    data_encoded.drop("No",inplace=True,axis=1)
    data.drop(["InternetService","PaymentMethod","Contract"],inplace=True,axis=1)
    data[data_encoded.columns]=data_encoded[data_encoded.columns]
    # now scaling the data only monthly charges and the total charges can be scaleed and tenure
    num_data=data[["tenure","MonthlyCharges","TotalCharges"]]
    scale=StandardScaler()
    data_scaled=scale.fit_transform(num_data)
    data_scaled=pd.DataFrame(data_scaled,columns=num_data.columns,index=num_data.index)
    data.drop(["tenure","MonthlyCharges","TotalCharges"],inplace=True,axis=1)
    data[num_data.columns]=data_scaled[num_data.columns]
    features=data.drop(["Churn","Unnamed: 0","customerID","MultipleLines","gender","PhoneService","DeviceProtection","StreamingTV","StreamingMovies"],axis=1)
    labels=data["Churn"]
    model=LogisticRegression(max_iter=1000)
    model.fit(features,labels)
    joblib.dump(model,"model.pkl")
else:
    encode=OneHotEncoder()
    data=pd.read_csv("test.csv")
    cat_data=data[['InternetService',"PaymentMethod","Contract"]]
    encoded_data=encode.fit_transform(cat_data)
    data_encoded=pd.DataFrame(encoded_data.toarray(),columns=['DSL', 'Fiber optic', 'No','Bank transfer (automatic)', 'Credit card (automatic)',
            'Electronic check', 'Mailed check','Month-to-month', 'One year', 'Two year'],index=data.index)
    data_encoded.drop("No",inplace=True,axis=1)
    data.drop(["InternetService","PaymentMethod","Contract"],inplace=True,axis=1)
    data[data_encoded.columns]=data_encoded[data_encoded.columns]
    # now scaling the data only monthly charges and the total charges can be scaleed and tenure
    num_data=data[["tenure","MonthlyCharges","TotalCharges"]]
    scale=StandardScaler()
    data_scaled=scale.fit_transform(num_data)
    data_scaled=pd.DataFrame(data_scaled,columns=num_data.columns,index=num_data.index)
    data.drop(["tenure","MonthlyCharges","TotalCharges"],inplace=True,axis=1)
    data[num_data.columns]=data_scaled[num_data.columns]
    features=data.drop(["Churn","Unnamed: 0","customerID","MultipleLines","gender","PhoneService","DeviceProtection","StreamingTV","StreamingMovies"],axis=1)
    labels=data["Churn"].copy()
    model=joblib.load("model.pkl")
    pre=model.predict(features).flatten()
    data["pre"]=pre
    data = data[[col for col in data.columns if col != "Churn"] + ["Churn"]]
    data.to_csv("last.csv")
