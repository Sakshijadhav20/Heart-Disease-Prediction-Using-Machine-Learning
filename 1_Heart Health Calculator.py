import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
import sklearn as sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

st.header("Heart Disease Risk Calculator")


##Loading Data
df = pd.read_csv('data/heart_2020_cleaned.csv')
newdf = df

#st.write("Our data is below:")
#st.write(newdf.head(20))
#st.write("We will now filter this data to focus on South Asian subjects for our model.")
#train = train[train["Race"] == "Asian"]
#st.write(train)

def user_input_features():
  
  st.write("**Please fill out the questionnaire below to see if you are at risk of diabetes:**")

  st.write("""**1. Enter BMI:**""") 
  BMI = st.slider('', 0.0, 110.0, 55.0)
  st.write("""**You selected this option:**""",BMI)

  st.write("""**2. Have you smoked over 100 cigarettes in your lifetime?**""") 
  smoke = st.selectbox("(Yes or No", ["Yes", "No"], key = "a")
  st.write("""**You selected this option:**""",smoke)

  st.write("""**3. Are you a heavy drinker? (>14 drinks per week for men, >7 drinks per week for women)**""") 
  alc = st.selectbox("(Yes or No", ["Yes", "No"], key = "b")
  st.write("""**You selected this option:**""",alc)

  st.write("""**4. Have you ever had a stroke?**""") 
  stroke = st.selectbox("(Yes or No", ["Yes", "No"], key = "c")
  st.write("""**You selected this option:**""", stroke)
  
  st.write("""**5. Of the last 30 days, how many would you consider \'bad\' days physically?**""") 
  physical = st.slider('', 0, 30, 15, key = "1")
  st.write("""**You selected this option:**""", physical)

  st.write("""**6. Of the last 30 days, how many would you consider \'bad\' days mentally?**""") 
  mental = st.slider('', 0, 30, 15, key = "2")
  st.write("""**You selected this option:**""", mental)

  st.write("""**7. Do you have difficulty climbing up stairs?**""") 
  climb = st.selectbox("(Yes or No", ["Yes", "No"], key = "d")
  st.write("""**You selected this option:**""", climb)

  st.write("""**8. What is your sex?**""") 
  sex = st.selectbox("(Male or Female", ["Male", "Female"], key = "e")
  st.write("""**You selected this option:**""", sex)

  st.write("""**9. Enter your age:**""") 
  age = st.slider('', 0, 100, 50)
  st.write("""**You selected this option:**""", age)

  age_cat= ""

  if age <= 29:
    age_cat = "25-29"
  elif age <= 34:
    age_cat = "30-34"
  elif age <= 39:
    age_cat = "35-39"
  elif age <= 44:
    age_cat = "40-44"
  elif age <= 49:
    age_cat = "45-49"
  elif age <= 54:
    age_cat = "50-54" 
  elif age <= 59:
    age_cat = "55-59"
  elif age <= 64:
    age_cat = "60-64"  
  elif age <= 69:
    age_cat = "65-69"  
  elif age <= 74:
    age_cat = "70-74" 
  elif age <= 79:
    age_cat = "75-79"  
  else:
    age_cat = "older than 80"
  
  st.write("""**10. Have you ever been told you are diabetic?**""") 
  diabetes = st.selectbox("(Yes or No)", ["Yes", "No"], key = "f")
  st.write("""**You selected this option:**""", diabetes)  
  
  st.write("""**11. Have you exercised in the past 30 days?**""") 
  exercise = st.selectbox("(Yes or No)", ["Yes", "No"], key = "g")
  st.write("""**You selected this option:**""", exercise)  

  st.write("""**12. How much do you sleep in a day (on avg):**""") 
  sleep = st.slider('', 0, 24, 12)
  st.write("""**You selected this option:**""", sleep) 
  
  st.write("""**13. How would you consider your general health?**""") 
  gen_health = st.selectbox("(Poor, Fair, Good, Very Good, Excellent)", ["Poor", "Fair", "Good", "Very Good", "Excellent"])
  st.write("""**You selected this option:**""", gen_health)

  st.write("""**14. Have you ever been told you have asthma?**""") 
  asthma = st.selectbox("(Yes or No)", ["Yes", "No"], key = "h")
  st.write("""**You selected this option:**""", asthma)  

  st.write("""**15. Have you ever been told you have kidney disease?**""") 
  kidney = st.selectbox("(Yes or No)", ["Yes", "No"], key = "i")
  st.write("""**You selected this option:**""", kidney)
  
  st.write("""**16. Have you ever been told you have skin cancer?**""") 
  cancer = st.selectbox("(Yes or No)", ["Yes", "No"], key = "j")
  st.write("""**You selected this option:**""", cancer)

  data = {'BMI': BMI, 'Smoking': smoke, 'AlcoholDrinking': alc, 'Stroke': stroke, 'PhysicalHealth': physical, 'MentalHealth': mental, 'DiffWalking': climb, 'Sex': sex, 'AgeCategory': age_cat, 'Race': 'Asian', 'Diabetic': diabetes, 'PhysicalActivity': exercise, 'GenHealth': gen_health,'SleepTime': sleep, 'Asthma': asthma, 'KidneyDisease': kidney, 'SkinCancer': cancer}
  features = pd.DataFrame(data, index=[0])
  st.subheader('Given Inputs : ')
  st.write(features)
  
  return features

user = user_input_features()

##Transform data

discrete = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
       'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth','Asthma', 'KidneyDisease', 'SkinCancer']     

discrete2 = ['Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
       'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth','Asthma', 'KidneyDisease', 'SkinCancer']     

from sklearn.preprocessing import OrdinalEncoder
enc = OrdinalEncoder()
enc.fit(newdf[discrete])
newdf[discrete] = enc.transform(newdf[discrete])

enc.fit(user[discrete2])
user[discrete2] = enc.transform(user[discrete2])

##Testing
##st.write(user)
##st.write(newdf)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score,recall_score
from sklearn.metrics import f1_score


##X_train, X_test, y_train, y_test=train_test_split(newdf,y,test_size=0.1,random_state=12)
param = newdf.iloc[:,0:17].values
target = newdf.iloc[:,[0]].values

#Testing
##st.write(param)
##st.write(target)


model = ExtraTreesClassifier()
model.fit(param, target)

prediction = model.predict(user)
st.subheader('Prediction using Logistic Regression:')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(user)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)

model = RandomForestClassifier()
model.fit(param, target)

prediction = model.predict(user)
st.subheader('Prediction using RandomForestClassifer:')
df1=pd.DataFrame(prediction,columns=['0'])
df1.loc[df1['0'] == 0, 'Chances of Heart Disease'] = 'No'
df1.loc[df1['0'] == 1, 'Chances of Heart Disease'] = 'Yes'
st.write(df1)

prediction_proba = model.predict_proba(user)
st.subheader('Prediction Probability in % :')
st.write(prediction_proba * 100)
