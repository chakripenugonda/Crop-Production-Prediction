import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("crop_production.csv")

df.dropna(inplace=True)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

from collections import Counter as c
print("before",c(df['Crop']))

#encoding for each column seperately

df['State_Name']=le.fit_transform(df['State_Name'])
df['District_Name']=le.fit_transform(df['District_Name'])
df['Season']=le.fit_transform(df['Season'])
df['Crop']=le.fit_transform(df['Crop'])

print("after",c(df['Crop']))


#independent variables
x=df.iloc[:,:6].values

#dependent variables
y=df.iloc[:,6:].values

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 44)



from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)
preds = model.predict(x_test)



from sklearn.metrics import r2_score
accuracy = r2_score(y_test,preds)
print("Accuracy when we predict using Randomn forest is ",accuracy)



import pickle
pickle.dump(model,open("model_rf.pkl","wb"))


model=pickle.load(open("model_rf.pkl","rb"))







