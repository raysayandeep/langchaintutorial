from fastapi import FastAPI
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple web server"
)

llm=Ollama(model="llama2")

prompt1=ChatPromptTemplate.from_template("Write me an essey about {topic} with 50 words")

add_routes(
    app,
    prompt1|llm,
    path="/essey"
)

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)
