# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:51:00 2023

@author: user
"""

import numpy as np
import pickle
import streamlit as st
#loaded the saved model
loaded_model=pickle.load(open("C:/Users/aryan/Downloads/TTrained_model.sav","rb"))


def churn_prediction(input_data):
    input_data=(128,25,10.0,3,2.70,265.1,110,45.07,197.4,99,16.78,244.7,91,11.01,1,1,0)
    input_data_as_numpy_array = np.asarray(input_data)
    #reshape the array as we predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)
    if prediction[0] == 0:
        return "NO"
    else:
        return "YES"


def main(): 
    #giving a tittle
    st.title("Churn prediction web app")
    #getting the data from the user
    account_length = st.text_input('account_length')
    voice_messages = st.text_input('voice_messages')
    intl_mins = st.text_input('intl_mins')
    intl_calls = st.text_input('intl_calls')	
    intl_charge = st.text_input('intl_charge')
    day_mins	= st.text_input('day_mins')
    day_calls = st.text_input('day_calls')
    day_charge = st.text_input('day_charge')
    eve_mins = st.text_input('eve_mins')
    eve_calls = st.text_input('eve_call')	
    eve_charge = st.text_input('eve_charge')
    night_mins = st.text_input('night_mins')
    night_calls = st.text_input('night_calls')
    night_charge = st.text_input('night_charge')
    customer_calls = st.text_input('customer_calls')
    voiceplan_yes = st.text_input('voiceplan_yes')
    intplan_yes = st.text_input('intplan_yes')
    
    
    #code for prediction  
    Churn=""
   
    #creating a button for prediction
    if st.button("Churn result"):
        Churn=churn_prediction([voice_messages,intl_mins,intl_calls,intl_charge,day_mins,day_calls,day_charge,eve_mins,eve_calls,eve_charge,night_mins,night_calls,night_charge,customer_calls,voiceplan_yes,intplan_yes])
       
        st.success(Churn)
       

if __name__=='__main__':
    main()
           
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    