# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.
2. Calculate the null values from dataframe and apply label encoder.
3. Apply decision tree classifier on the dataframe.
4. Obtain the value of accuracy and data prediction.

## Program:

Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: GOKUL ,
RegisterNumber:  212221220013

```py
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```




## Output:
### data.head(): 
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/d1f9945e-2801-46d0-a1ca-155e274ae349)


### data.info():
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/4574e54a-0bbc-404c-a48b-e022c323281f)


### is null() and sum():
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/f0ec4771-145b-4b2d-88b4-9402155e66b7)


### data value counts():
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/4f2cadb3-8b38-446f-a710-4a44c531a803)


### data.head() for salary:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/17af61aa-d656-486b-bd25-0eba92fbaa7b)


### x.head():
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/59ccdd45-7d7f-4c6c-9200-a17be649cfd9)


### ACCURACY VALUE:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/64886b22-d78b-4111-9c27-07228b9c501c)


### DATA PREDICTION:
![image](https://github.com/babavoss05/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/103019882/d67a1921-f4b6-487a-aa24-992b0b140327)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
