# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook=
## Algorithm
```
1.Import the required packages.
2.Print the present data and placement data and salary data.
3.Using logistic regression find the predicted values of accuracy confusion matrices.
4.Display the results.
```
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: LOGA MITHRA R
RegisterNumber:  212223100027
```
```

import pandas as pd
df=pd.read_csv("Placement_Data(1).csv")
df.head()

df1=df.copy()
df1=df1.drop(["sl_no","salary"],axis=1)
df1.head()

df1.isnull().sum()

df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x=df1.iloc[:,:-1]
x

y=df1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
## Output:
## Placement Data:
![image](https://github.com/user-attachments/assets/e7ca4571-6c5c-4a5e-b653-ff240095ec6d)

## Checking the null() function:
![image](https://github.com/user-attachments/assets/ce9c16d1-2def-4dfc-9c06-02e07957b7ec)

## Data Duplicate:
![image](https://github.com/user-attachments/assets/5aa90d84-4cea-491f-88e8-89d82a514722)

## Print Data:
![image](https://github.com/user-attachments/assets/50de15f3-63a1-4fff-943f-5b92c4d375b3)

## Y_prediction array:
![image](https://github.com/user-attachments/assets/4adc16cb-441a-447b-a549-807549a54c5f)

## Accuracy value:
![image](https://github.com/user-attachments/assets/12632ddc-6127-4db4-8257-7da3758bcd9d)

## Confusion array:
![image](https://github.com/user-attachments/assets/4e4bbde1-d456-4775-97a4-8fc36e38ae12)

## Classification Report:
![image](https://github.com/user-attachments/assets/a0d2886f-8c4e-420c-81fc-0941d0a12d0e)

## Prediction of LR:
![image](https://github.com/user-attachments/assets/61598743-b03d-478f-8b4f-735501bb5c5e)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
