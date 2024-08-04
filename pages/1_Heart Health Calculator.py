import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder

st.header("Heart Disease Risk Calculator")

# Load Data
df = pd.read_csv('C:/Users/SAKSHI/OneDrive/Pictures/HealHeart/HealHeart/data/heart_2020_cleaned.csv')
newdf = df

def user_input_features():
    st.write("**Please fill out the questionnaire below to see if you are at risk of heart disease:**")

    BMI = st.slider('Enter BMI:', 0.0, 110.0, 55.0)
    smoke = st.selectbox('Have you smoked over 100 cigarettes in your lifetime?', ["Yes", "No"])
    alc = st.selectbox('Are you a heavy drinker? (>14 drinks per week for men, >7 drinks per week for women)', ["Yes", "No"])
    stroke = st.selectbox('Have you ever had a stroke?', ["Yes", "No"])
    physical = st.slider('Of the last 30 days, how many would you consider "bad" days physically?', 0, 30, 15)
    mental = st.slider('Of the last 30 days, how many would you consider "bad" days mentally?', 0, 30, 15)
    climb = st.selectbox('Do you have difficulty climbing up stairs?', ["Yes", "No"])
    sex = st.selectbox('What is your sex?', ["Male", "Female"])
    age = st.slider('Enter your age:', 0, 100, 50)

    # Categorize age
    age_cat = ""
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
        age_cat = "80+"

    diabetes = st.selectbox('Have you ever been told you are diabetic?', ["Yes", "No"])
    exercise = st.selectbox('Have you exercised in the past 30 days?', ["Yes", "No"])
    sleep = st.slider('How much do you sleep in a day (on avg):', 0, 24, 12)
    gen_health = st.selectbox('How would you consider your general health?', ["Poor", "Fair", "Good", "Very Good", "Excellent"])
    asthma = st.selectbox('Have you ever been told you have asthma?', ["Yes", "No"])
    kidney = st.selectbox('Have you ever been told you have kidney disease?', ["Yes", "No"])
    cancer = st.selectbox('Have you ever been told you have skin cancer?', ["Yes", "No"])

    data = {'BMI': BMI, 'Smoking': smoke, 'AlcoholDrinking': alc, 'Stroke': stroke, 'PhysicalHealth': physical,
            'MentalHealth': mental, 'DiffWalking': climb, 'Sex': sex, 'AgeCategory': age_cat, 'Race': 'Asian',
            'Diabetic': diabetes, 'PhysicalActivity': exercise, 'GenHealth': gen_health, 'Asthma': asthma,
            'KidneyDisease': kidney, 'SkinCancer': cancer}
    features = pd.DataFrame(data, index=[0])
    st.subheader('Given Inputs:')
    st.write(features)

    return features

user = user_input_features()

# Define feature columns (excluding the target variable 'HeartDisease')
discrete = ['BMI', 'Smoking', 'AlcoholDrinking', 'Stroke', 'PhysicalHealth', 'MentalHealth', 'DiffWalking',
            'Sex', 'AgeCategory', 'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth', 'Asthma', 'KidneyDisease',
            'SkinCancer']

# Encode categorical features
enc = OrdinalEncoder()
newdf[discrete] = enc.fit_transform(newdf[discrete])
user[discrete] = enc.transform(user[discrete])

# Separate features and target variable
X = newdf[discrete]
y = newdf['HeartDisease']  # Assuming 'HeartDisease' is the target variable

# Logistic Regression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X, y)
log_proba = log_model.predict_proba(user[discrete])
log_predictions = (log_proba[:, 1] > 0.5).astype(int)  # 1 indicates 'Yes'
df_log = pd.DataFrame({'Heart Disease': log_predictions})
df_log['Chances of Heart Disease'] = np.where(df_log['Heart Disease'] == 0, 'No', 'Yes')
st.subheader('Logistic Regression Prediction:')
st.write(df_log)
st.subheader('Logistic Regression Prediction Probability (%):')
st.write(log_proba * 100)

# Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X, y)
rf_proba = rf_model.predict_proba(user[discrete])
rf_predictions = (rf_proba[:, 1] > 0.5).astype(int)  # 1 indicates 'Yes'
df_rf = pd.DataFrame({'Heart Disease': rf_predictions})
df_rf['Chances of Heart Disease'] = np.where(df_rf['Heart Disease'] == 0, 'No', 'Yes')
st.subheader('Random Forest Prediction:')
st.write(df_rf)
st.subheader('Random Forest Prediction Probability (%):')
st.write(rf_proba * 100)

