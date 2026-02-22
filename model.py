import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score

# this file is fro model traing which model is working which to select. features we are droppping are helpfull or not
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
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

roc_auc_scores = cross_val_score(
    model,      # pipeline
    features, labels,
    cv=cv,
    scoring="roc_auc"
)

print("ROC-AUC scores:", roc_auc_scores)
print("Mean ROC-AUC:", roc_auc_scores.mean())