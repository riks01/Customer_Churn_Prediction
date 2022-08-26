import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder

m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: rgb(30, 200, 49);font-size:15px;height:3em;width:20em;
}
</style>""", unsafe_allow_html=True)

st.title('CUSTOMER CHURN ANALYSIS')

st.sidebar.header('User Input Parameters')

def user_input_features():
    sc = st.sidebar.selectbox('Senior Citizen',(0,1))
    pr = st.sidebar.selectbox('Partner',('Yes','No'))
    dep = st.sidebar.selectbox('Dependents',('Yes','No'))
    ten = st.slider("Tenure", min_value=0, max_value=75, step=1)
    # ml = st.sidebar.selectbox('MultipleLines', ('No phone service', 'No', 'Yes'))
    # isr = st.sidebar.selectbox('InternetService', ('DSL', 'Fiber optic', 'No'))
    osr = st.sidebar.selectbox('OnlineSecurity', ('No', 'Yes', 'No internet service'))
    ob = st.sidebar.selectbox('OnlineBackup', ('No', 'Yes', 'No internet service'))

    dp = st.sidebar.selectbox('DeviceProtection', ('No', 'Yes', 'No internet service'))
    stv = st.sidebar.selectbox('StreamingTV', ('No', 'Yes', 'No internet service'))
    sms = st.sidebar.selectbox('StreamingMovies', ('No', 'Yes', 'No internet service'))

    ts = st.sidebar.selectbox('TechSupport', ('No', 'Yes', 'No internet service'))

    cr = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))

    pb = st.sidebar.selectbox('PaperlessBilling',('Yes', 'No'))
    pm = st.sidebar.selectbox('PaymentMethod',('Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'))

    mc = st.sidebar.number_input("Insert the MonthlyCharges", min_value=10, max_value=1000, step=1)
    tc = st.sidebar.number_input("Insert TotalCharges", min_value=10, max_value=1000, step=1)

    data = {'SeniorCitizen': sc,
            'Partner': pr,
            'Dependents': dep,
            'tenure': ten,
            'OnlineSecurity': osr,
            'OnlineBackup': ob,
            'DeviceProtection': dp,
            'TechSupport': ts,
            'StreamingTV': stv,
            'StreamingMovies': sms,
            'Contract': cr,
            'PaperlessBilling': pb,
            'PaymentMethod': pm,
            'MonthlyCharges': mc,
            'TotalCharges': tc}

    features = pd.DataFrame(data,index = [0])
    return features


X = user_input_features()
st.subheader('User Input parameters')
st.write(X)

# Convert all categorical columns to numerical categorical columns
cols = X.select_dtypes('object').columns
to_fit = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
enc = OneHotEncoder(handle_unknown='ignore')
cat_data = enc.fit(to_fit[cols])
cat_data = pd.DataFrame(enc.transform(X[cols]).toarray())
cat_data.columns = enc.get_feature_names()

num_data = X.select_dtypes('number')

#concatenating numeric and categorical data
x = pd.concat([cat_data,num_data], axis=1)

if st.button('predict'):
    with open(file="model.sav",mode="rb") as f1:
        model = pickle.load(f1)

    prediction = model.predict(x)
    prediction_proba = np.round(max(model.predict_proba(x)[0]),2)
    print(prediction)
    print(prediction_proba)
    st.subheader('Predicted Result')
    st.write('Yes' if prediction == 1 else 'No')

    st.subheader('Prediction Probability')

    st.write(prediction_proba)





