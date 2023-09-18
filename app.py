import streamlit as st
import pickle
import numpy as np
import pandas as pd

st.set_page_config(
    page_title='Titanic Survivor Prediction',
    page_icon=':ship:'
)

st.title('Titanic Survivor Prediction Model Deployment')

if 'model' not in st.session_state:
    model = pickle.load(open('data/model.sav','rb'))
    st.session_state['model'] = model

pclass = st.selectbox('Pclass',
                      (3, 2, 1))
sex = st.selectbox('sex',
                      ('male', 'female'))
age = st.number_input('age')
sibsp = st.number_input('sibsp')
parch = st.number_input('parch')
fare = st.number_input('fare')
embarked = st.selectbox('embarked',
                      ('Q', 'S', 'C'))
class_ticket = st.selectbox('class',
                      ('First', 'Second', 'Third'))
who = st.selectbox('who',
                      ('man', 'woman'))
adult_male = st.selectbox('adult male',
                      (True, False))
alone = st.selectbox('alone',
                      (True, False))

if st.button('Model Predict'):
    data = pd.DataFrame({
        'pclass': [pclass],
        'sex': [sex],
        'age': [age],
        'sibsp': [sibsp],
        'parch': [parch],
        'fare': [fare],
        'embarked': [embarked],
        'class': [class_ticket],
        'who': [who],
        'adult_male': [adult_male],
        'alone': [alone]
    })
    result = st.session_state['model'].predict(data)
    if result[0] == 0:
        st.write(f'passanger not survived')
    else:
        st.write(f'passanger survived')

else:
    st.write('Please input the feature above to start modelling')

