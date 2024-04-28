import requests
import streamlit as st

#streamlit

st.title('Langchain RAG Demo with Llama3 API')


with st.form("my-form"):
    url = "http://localhost:8000/upload"
    payload = ''
    uploaded_file = st.file_uploader("Choose a file")
    input_text=st.text_input("Ask a question:","Who is the Author of this Paper?")
    submitted = st.form_submit_button("Submit")
    if submitted and input_text and uploaded_file:
        payload = {"file": uploaded_file.name,"query":input_text}
        with st.spinner("Generating Model Response:"):
            response = requests.post(url, params=payload, files={"file": uploaded_file.getvalue()}).json()
            st.write("Model Response:")
            st.write(response)