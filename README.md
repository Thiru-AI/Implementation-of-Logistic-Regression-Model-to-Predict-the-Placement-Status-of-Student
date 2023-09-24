# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Load the dataset and check for null data values and duplicate data values in the dataframe.
3. Import label encoder from sklearn.preprocessing to encode the dataset.
4. Apply Logistic Regression on to the model.
5. Predict the y values.
6. Calculate the Accuracy,Confusion and Classsification report.

## Program:
~~~
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Thirugnanamoorthi G
RegisterNumber: 212221230117
~~~

~~~
import pandas as pd
data=pd.read_csv('Placement_Data.csv') 
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state =0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
~~~

## Output:
### 1.Placement data
![image](https://user-images.githubusercontent.com/93587823/236502062-fe7dc648-dba9-4768-bd03-e4175911f131.png)

### 2.Salary data
![image](https://user-images.githubusercontent.com/93587823/236502517-e4d9e285-facb-49bc-98a7-381a77f71587.png)

### 3.Checking the null() function
![image](https://user-images.githubusercontent.com/93587823/236502575-adb28ef4-9e96-4fa7-8e9a-feec237d32f9.png)

### 4. Data Duplicate

![image](https://user-images.githubusercontent.com/93587823/236502665-b592273d-36c2-48c6-9758-b755b853ff17.png)
### 5. Print data
![image](https://user-images.githubusercontent.com/93587823/236502762-d6aa4656-65d3-4130-b474-9a259ed51c03.png)

### 6. Data-status
![image](https://user-images.githubusercontent.com/93587823/236502828-c76a3eb3-ac28-4081-8bf5-d6c51a8f727f.png)

### 7. y_prediction array
![image](https://user-images.githubusercontent.com/93587823/236502880-23c95889-c379-45ce-b154-813f7870781a.png)

### 8.Accuracy value
![image](https://user-images.githubusercontent.com/93587823/236502910-c414458a-bd39-4715-b1cd-c15a9793ff76.png)

### 9. Confusion array
![image](https://user-images.githubusercontent.com/93587823/236502992-8466e2f3-330d-4301-b97d-756c081ca147.png)

### 10. Classification report
![image](https://user-images.githubusercontent.com/93587823/236503089-976a0de5-7894-4309-a6a6-51f2b6a0f33c.png)
### 11.Prediction of LR
![image](https://user-images.githubusercontent.com/93587823/236503185-ed221ba7-f368-4396-b1ce-2e7b11eebb9b.png)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
