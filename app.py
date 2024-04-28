from fastapi import FastAPI, UploadFile
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from langserve import add_routes
import uvicorn
import os
from dotenv import load_dotenv
from chain import get_chain

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")

app=FastAPI(
    title="Langchain Server",
    version="1.0",
    description="A simple web server"
)

@app.post("/upload")
def upload_file(file: UploadFile,query):
    file_location = f"samplefiles/{file.filename}"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    #return {"info": f"file '{file.filename}' saved at '{file_location}'"}
    response = get_chain(file_location,query)
    print(response)
    os.remove(file_location)   
    return response

llm=Ollama(model="llama3",stop=['<|eot_id|>'])

prompt1=ChatPromptTemplate.from_template("Write me an essay about {topic} with 200 words")

add_routes(
    app,
    prompt1|llm,
    path="/essey"
)

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8000)
