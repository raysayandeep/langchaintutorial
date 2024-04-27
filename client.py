import requests
import streamlit as st

def get_ollama_response(input_text):
    response=requests.post(
        "http://localhost:8000/essey/invoke",
        json={'input':{'topic':input_text}}
    )
    return response.json()['output']

def get_rag_response(uploaded_file):
    response=requests.post("http://localhost:8000/upload"  
    )

#streamlit

st.title('Langchain Demo with Llama3 API')
input_text=st.text_input("Write an essay on")

if input_text:
    with st.spinner("Generating Model Response:"):
        st.write("Model Response:")
        st.write(get_ollama_response(input_text))

uploaded_file = st.file_uploader("Choose a file")
url = "http://localhost:8000/upload"
if st.button("Upload PDF"):
    payload = {"file": uploaded_file.name}
    filename = uploaded_file.name
    with st.spinner("Generating Model Response:"):
        response = requests.post(url, params=payload, files={"file": uploaded_file.getvalue()}).json()
        st.write("Model Response:")
        st.write(response)