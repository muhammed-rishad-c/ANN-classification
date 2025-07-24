import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
import pickle


model=tf.keras.models.load_model('model/model.h5')

with open('pickle-file/label_encoder_gender.pkl','rb') as file:
    label_encoder=pickle.load(file)
   
with open('pickle-file/one_hot_encoder_geo.pkl','rb') as file:
    onehot_encoder=pickle.load(file) 
    
with open('pickle-file/scaler.pkl','rb') as file:
    scaler=pickle.load(file)
    

st.title('customer churn prediction')
# User input
geography = st.selectbox('Geography', onehot_encoder.categories_[0])
gender = st.selectbox('Gender', label_encoder.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])

input_data=pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender":[label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded=onehot_encoder.transform([[geography]]).toarray()
one_hot_geo_df=pd.DataFrame(geo_encoded,columns=onehot_encoder.get_feature_names_out(['Geography']))

input_df=pd.concat((input_data.reset_index(drop=True),one_hot_geo_df),axis=1)

input_scaled_data=scaler.transform(input_df)

prediction=model.predict(input_scaled_data)

prediction_proba=prediction[0][0]

if prediction_proba >0.5:
    st.write('the customer is likely to churn')
    
else:
    st.write("customer is not going to churn")