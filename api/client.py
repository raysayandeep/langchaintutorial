import requests
import streamlit as st

def get_ollama_response(input_text):
    response=requests.post(
        "http://localhost:8000/essey/invoke",
        json={'input':{'topic':input_text}}
    )
    return response.json()['output']

#streamlit

st.title('Langchain Demo with Llama3 API')
input_text=st.text_input("Write an essay on")

if input_text:
    with st.spinner("Generating Model Response:"):
        st.write("Model Response:")
        st.write(get_ollama_response(input_text))