import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(
    page_title= "Homepage",
    layout="wide")


sns.set(rc={'axes.facecolor':'#F5F6F4', 'figure.facecolor':'#F5F6F4'})



col1, col2 = st.columns([1,6])

with col1:
   st.image("images/HeartGif.gif")

with col2:
    new_title = '<p style="font-family:sans-serif; color:#1E000E; font-size: 70px;">Heart Health Model</p>'
    st.markdown(new_title, unsafe_allow_html=True)

    
st.write("Heart disease is a leading cause of death and disability in the South Asian community. Our app aims to provide accurate and reliable information on the risk of heart disease in South Asians, empowering individuals to make informed decisions about their health and take proactive steps towards maintaining a healthy heart. By raising awareness and promoting prevention, we hope to reduce the burden of heart disease in the South Asian community and improve the overall health and well-being of our community.")

st.write("The graph below illustrates the prevalence of heart disease among different racial and ethnic groups. As we can see, South Asians are disproportionately affected by heart disease, with some of the highest rates among the groups depicted. These disparities highlight the need for targeted prevention and treatment efforts within the South Asian community in order to reduce the burden of heart disease and improve overall health outcomes.")

df = pd.read_csv('data/DeathsbyRace.csv')
fig = plt.figure(figsize=(5, 5))
sns.barplot(data=df, y="Race", x="Percent(%) of Deaths", color="#C27070")
st.pyplot(fig)

st.write("In order to most effectively use our model, we recommend you fill out all of your information and then compare to the provided mean values. By doing so, you will able to identify what specifically is the variable that is leading to heart disease for you. Recommendations for fixing these habits, as well as an explanation of the variables, are provided in the \"Health Recommendations\" page")
