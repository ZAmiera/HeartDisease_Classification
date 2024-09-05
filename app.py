# from sklearn.datasets import load_iris
import pandas as pd
import streamlit as st 
import pickle
import numpy as np

HeartDisease = ['Heart Disease', 'Normal'] 
with open('model_pipeline.pkl', 'rb') as model_file:
    model = pickle.load(model_file)


def main():

    # Creating Sidebar for inputs
    st.sidebar.title("Inputs")
    Age             = st.sidebar.slider("Age",28,77,50)
    Sex             = st.sidebar.slider("Sex (Male: 1, Female: 0)",0,1,0)
    ChestPainType   = st.sidebar.slider("Chest Pain Type (Typical Angina: 0, Atypical Angina: 1, Non-Anginal Pain: 2, Asymptomatic: 3)",0,3,1)
    Cholesterol     = st.sidebar.slider("Serum Cholesterol (mm/dl)",0.1,2.5,0.2)
    FastingBS       = st.sidebar.slider("Fasting Blood Sugar (1: if FastingBS > 120 mg/dl, 0: otherwise)",0,1,0)
    MaxHR           = st.sidebar.slider("Maximum Heart Rate",4.3,7.9,5.0)
    ExerciseAngina  = st.sidebar.slider("Exercise-Induced Angina (Yes: 1, No: 0)",0,1,0)
    Oldpeak         = st.sidebar.slider("Oldpeak",-2.6,6.2,1.0)
    ST_Slope        = st.sidebar.slider("Slope of the peak exercise ST segment ('Downsloping': 2, 'Flat': 1, 'Upsloping': 0)",0,2,0)

    # Getting Prediction from model
    inp = np.array([Age,Sex,ChestPainType,Cholesterol,FastingBS,MaxHR,ExerciseAngina,Oldpeak,ST_Slope])
    inp = np.expand_dims(inp,axis=0)
    prediction = model.predict_proba(inp)

    # Main page
    st.title("Heart Disease Classification")
    st.write("This app classifies weather someone have a heart disease or not")

    ## Show Results when prediction is done
    if prediction.any():
        st.write('''
        ## Results
        Following is the probability of each class
        ''')
        
        df = pd.DataFrame(prediction, index = ['result'], columns=HeartDisease)
        st.dataframe(df)
        result = HeartDisease[np.argmax(prediction)]
        if result == "Normal":
            st.write("**This person DO NOT HAVE a heart disease**")
        else:
            st.write("**This person HAVE a heart disease**")

        

    

if __name__ == "__main__":
    main()