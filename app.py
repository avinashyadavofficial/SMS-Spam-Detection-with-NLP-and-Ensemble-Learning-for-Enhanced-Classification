import streamlit as st
import joblib
import json
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained stacking model
stacking_model = joblib.load("stacking_model.pkl")

# Load the tokenizer from the saved JSON file
with open("tokenizer.json", "r") as json_file:
    tokenizer_data = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_data)

# Function to preprocess user input
def preprocess_message(message):
    message = message.lower()
    # Convert the message into a sequence of integers
    sequence = tokenizer.texts_to_sequences([message])
    # Pad the sequence to match training input length
    padded_sequence = pad_sequences(sequence, maxlen=50)
    return padded_sequence

# Define the Streamlit app
def app():
    st.title("SMS Spam Detector")
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F5F5F5;
            padding: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # User input
    message = st.text_input("Enter a message to classify:")
    
    if message:
        # Preprocess input and make a prediction
        processed_message = preprocess_message(message)
        prediction = stacking_model.predict(processed_message)
        
        # Display the result
        if prediction > 0.5:
            st.write("🔴 This message is spam with a probability of {:.2f}%.".format(prediction[0][0] * 100))
        else:
            st.write("🟢 This message is not spam (ham) with a probability of {:.2f}%.".format((1 - prediction[0][0]) * 100))

if __name__ == "__main__":
    app()
