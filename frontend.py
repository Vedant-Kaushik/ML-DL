import streamlit as st
import requests

API_URL = "http://localhost:8000/predict" 

st.title("Joke Generator")

st.markdown("Enter your topic below:")

# Input fields

text = st.text_input("Topic to Joke", value="boy")

if st.button("Generate"):
    input_data = {
        "text": text
    }

    try:
        response = requests.post(API_URL, json=input_data)
        if response.status_code == 200:
            result = response.json()
            st.success(f"Generated Joke: **{result['generated_joke']}**")
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to the FastAPI server. Make sure it's running on port 8000.")
