import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import time

# https://docs.streamlit.io/tutorial/index.html
# https://datasciencechalktalk.com/2019/10/22/building-machine-learning-apps-with-streamlit/
# https://docs.streamlit.io/api.html#magic-commands

# Data
st.title("Iris Data Set")
df = pd.read_csv("iris.csv")
start = time.time()

if st.checkbox("Show Dataframe"):
    species = st.write('iris labels',df['target'].unique())
    st.markdown('These are the columns')
    st.write(df.columns)

    st.markdown('Iris DataFrame')
    st.write(df)

st.subheader("Scatter Plot")
xaxis = st.selectbox('X axis:',df.columns[:-1])
yaxis = st.selectbox('Y axis:',df.columns[:-1])

if st.checkbox("Display Scatter Plot"):
    st.markdown('Ploty Express')
    fig = px.scatter(df,x = xaxis,y = yaxis,color = 'target')
    st.plotly_chart(fig)

st.subheader('Histogram')
feature = st.selectbox('Which Feature?',df.columns[:-1])

if st.checkbox("Display Histogram"):
    fig2 = px.histogram(df,x =feature,color = 'target')
    st.plotly_chart(fig2)


st.header('Machine Learning Models')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC

features = df[df.columns[:-1]]
st.markdown('Samples_Features DF')
st.write(features.head(5))

labels = df.drop(features,axis = 1)

x_train,x_test,y_train,y_test = train_test_split(features,labels,train_size =0.8, random_state = 101)

algs = ['None','Decision Tree','Support Vector Machine']
classifier = st.selectbox('Which Algorithm to choose?',algs)

if st.button('Run Selected Machine Learning Model'):
    if classifier == 'None':
        st.warning('Please Select a Model {}'.format(algs[1:]))

    elif classifier == 'Decision Tree':
        train_time = time.time()
        tree = DecisionTreeClassifier()
        tree.fit(x_train,y_train)
        acc = tree.score(x_test,y_test)
        st.write('Accuracy: ', acc)

        predictions = tree.predict(x_test)
        confusion = confusion_matrix(y_test,predictions)
        st.write('Confusion Matrix: ',confusion)
        ed_time = time.time()
        st.write('Time Spent to Complete Result: ',ed_time-train_time)


    elif classifier == 'Support Vector Machine':
        train_time = time.time()
        svm = SVC()
        svm.fit(x_train,y_train)
        acc = svm.score(x_test,y_test)
        st.write('Accuarcy: ', acc)

        predictions = svm.predict(x_test)
        confusion = confusion_matrix(y_test,predictions)
        st.write('Confusion Matrix:', confusion)
        ed_time = time.time()
        st.write('Time Spent to Complete Result: ',ed_time-train_time)


ed = time.time()
st.write('Time Spent to Run This App: ',ed-start)
