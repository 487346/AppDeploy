#!/usr/bin/env python
# coding: utf-8

# In[ ]:
import streamlit as st
import gender_guesser.detector as gender
from transformers import pipeline

# Create a gender detector instance
detector = gender.Detector()

# Function to predict gender using gender-guesser
def predict_gender(name):
    gender_result = detector.get_gender(name)
    if gender_result == 'male' or gender_result == 'mostly_male':
        return 'Male'
    elif gender_result == 'female' or gender_result == 'mostly_female':
        return 'Female'
    else:
        return 'Unknown'

# Function to predict age bracket based on name using a simple approach
def predict_age_bracket(age):
    age_brackets = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+']
    
    if age < 10:
        return age_brackets[0]
    elif age < 20:
        return age_brackets[1]
    elif age < 30:
        return age_brackets[2]
    elif age < 40:
        return age_brackets[3]
    elif age < 50:
        return age_brackets[4]
    elif age < 60:
        return age_brackets[5]
    elif age < 70:
        return age_brackets[6]
    elif age < 80:
        return age_brackets[7]
    else:
        return age_brackets[8]

# Create a Streamlit app
def main():
    st.title('Age and Gender Prediction')

    st.write("This app predicts gender and age brackets based on input. Enter a name and age to get the prediction.")

    # Input fields
    name = st.text_input('Enter your name:')
    age = st.number_input('Enter your age:', min_value=0, max_value=120, step=1)

    if st.button('Predict'):
        if name:
            gender_prediction = predict_gender(name)
            st.write(f"Predicted Gender: {gender_prediction}")
        else:
            st.write("Please enter a valid name.")
        
        if age is not None:
            age_prediction = predict_age_bracket(age)
            st.write(f"Predicted Age Bracket: {age_prediction}")
        else:
            st.write("Please enter a valid age.")
        
if __name__ == "__main__":
    main()


# In[ ]:
