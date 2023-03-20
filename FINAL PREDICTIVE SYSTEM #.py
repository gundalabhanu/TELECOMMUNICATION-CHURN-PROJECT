# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 14:45:06 2023

@author: user
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
#loaded the saved model
loaded_model=pickle.load(open("C:/Users/aryan/Downloads/TTrained_model.sav","rb"))

input_data=(128,25,10.0,3,2.70,265.1,110,45.07,197.4,99,16.78,244.7,91,11.01,1,1,0)
#changing the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we predicting for one instance
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if(prediction[0]==0):
    print("NO")
else:
    print("YES")