import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import streamlit as st
import sklearn as sklearn

sns.set(rc={'axes.facecolor':'#F5F6F4', 'figure.facecolor':'#F5F6F4'})


#st.header("Exploratory Data Analytics")

st.write("On this page we will walkthrough our data analytics process as we worked towards building our model for the Asian Heart Health Model.")

st.write("First we understood our model in the context of all races; below is a overview of the data we used:")

##Loading Data
df = pd.read_csv('data/heart_2020_cleaned.csv')
#st.write(df.sample(50))


st.write("Next, we sorted our data into continuous and discrete variables. This laid the groundwork for any future transformations we had to conduct.")
col1, col2 = st.columns(2)

continous = ['BMI', 'PhysicalHealth', 'MentalHealth', 'SleepTime']
discrete = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'Sex', 'AgeCategory',
       'Race', 'Diabetic', 'PhysicalActivity', 'GenHealth','Asthma', 'KidneyDisease', 'SkinCancer']
## Update the lists
with col1:
    st.header("Discrete")
    s = ""
    for i in discrete:
        s += "- " + i + "\n" 
    st.markdown(s) 
   


with col2:
    st.header("Continuous")
    s = ""
    for i in continous:
        s += "- " + i + "\n" 
    st.markdown(s) 


st.write("We then determined if there was any null data within our data")
code = '''df.isnull().sum()
    df.dropna(axis = 0, inplace = True)
    df.isnull().sum().sum()  '''
st.code(code, language='python')

null = df.isnull().sum().sum()
st.write("Null Values = ", str(null))

column = ['Race','BMI','Smoking','AlcoholDrinking','Stroke','PhysicalHealth','MentalHealth','Sex','Diabetic','PhysicalActivity','SleepTime','AgeCategory','Asthma','KidneyDisease']
st.write("Now that we were confident in our data we started to do a biological analysis of what factors do and do not matter in order to have the most efficient and effecitive model. We determined the factors we needed were: ", column)

st.header('Feature Engineering')
st.write("In this section we will be transforming our data in order to build a model as well as analyze our variables using statistical measures.")
st.text("\n")
st.write("First lets turn all categorical variables into booleans or numerical ranges")
st.write("Here's an example using alcohol, below you can see we turned every input that was \"Yes\" into a 1 and every input that was a \"No\" into a 0." )
code = '''df.AlcoholDrinking.replace(('Yes','No'), (1,0), inplace = True)'''
st.code(code, language = 'python')
df.Smoking.replace(('Yes','No'), (1,0), inplace = True)
df.HeartDisease.replace(('Yes','No'), (1,0), inplace = True)
df.AlcoholDrinking.replace(('Yes','No'), (1,0), inplace = True)
df.Stroke.replace(('Yes', "No"), (1,0), inplace = True)
df.Sex.replace(('Male','Female'), (1,0), inplace = True)
df.Asthma.replace(('Yes','No'), (1,0), inplace = True)
df.Diabetic.replace(('Yes','No','No, borderline diabetes','Yes (during pregnancy)'), (1,0,0,0), inplace = True)
df.PhysicalActivity.replace(('Yes','No'), (1,0), inplace = True)
df.KidneyDisease.replace(('Yes','No'), (1,0), inplace = True)
df.AgeCategory.replace(('18-24','25-29','30-34','35-39','40-44','45-49','50-54',
                          '55-59','60-64','65-69','70-74','75-79','80 or older'),
                         (0,0,1,1,1,2,2,2,3,3,4,5,6), inplace = True)
df.Race.replace(('White', 'Black', 'Asian', 'American Indian/Alaskan Native', 'Other', 'Hispanic'),
                          (1,2,3,1,1,2), inplace = True)

st.write('Let\'s look at the data now coverted into numbers and only containing the columns we need.')
data=df[column]
st.write(data.sample(50))

st.write('Now lets seperate the data into a train and test data set.')
X = data[column]
st.write(X.head())

Y = df.HeartDisease
st.write('Y Value counts: ', Y.value_counts())

st.write('We can see from the value counts that the data is not balanced between cases of heart disease and cases of no heart disease.')
st.write('This could cause potential errors; to avoid that we can implement SMOTE which will assist with the oversampling.')
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 0)
sm.fit(X,Y)
x_resem, y_resem = sm.fit_resample(X, Y)

code = '''from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 0)
sm.fit(X,Y)
x_resem, y_resem = sm.fit_resample(X, Y)'''
st.code(code, language='python')

st.write('Lets see the distribution of data now:', y_resem.value_counts())


st.header('Model Building')

st.write('We build our model using Scikit-Learn, the model we chose to use was a DecisionTreeClassifier.')
st.markdown("\n")
st.write('We first split the data into xtrain, xtest, ytrain, ytest using a test size of 0.2.')
with st.echo():
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest = train_test_split(x_resem, y_resem, test_size = 0.2, random_state = 0)
st.write("Next, we built our model by importing DecisionTreeClassifier and fitting it to our data")

with st.echo():
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier()
    model.fit(xtrain, ytrain)
    model.score(xtest,ytest)

st.write('After fitting the model to our data we ran a couple tests on effectivness of the model. Below, you can see a confusion matrix as well as tests on the accuracy and precision of the model')
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score, recall_score

st.write("Confusion Matrix")
ypred = model.predict(xtest)
matrix = confusion_matrix(ytest,ypred)

# Convert the confusion matrix into a Pandas DataFrame
matrix_df = pd.DataFrame(matrix, index=['True Neg','True Pos'], columns=['Pred Neg','Pred Pos'])


# Use the seaborn heatmap function to plot the confusion matrix
plot = sns.heatmap(matrix_df, annot=True, fmt='d', cmap='Reds')

# plot is being inconsistent so I uploaded an image of the plot
st.image('images/plot.png')

st.write('Accuracy Scores')
# Calculate the evaluation scores
accuracy = accuracy_score(ytest, ypred)
precision = precision_score(ytest, ypred)
recall = recall_score(ytest, ypred)
f1 = f1_score(ytest, ypred)

# Create a dictionary with the evaluation scores
scores = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}

# Convert the dictionary into a Pandas DataFrame
scores_df = pd.DataFrame(scores, index=['Score'])

scores_df = scores_df.T
st.table(scores_df)
