from langchain_community.llms import Ollama
from langchain_community.llms import Bedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser

from rag.faissvector import loadVectorDB
from rag.embeddings import initiateOllamaEmbedding
from rag.prompts import generatePrompt

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

def localLlama3():
    local_llama3=Ollama(model="Llama3",stop=['<|eot_id|>'])
    return local_llama3

def localLlama2():
    local_llama2=Ollama(model="Llama3")
    return local_llama2

#Example: Bedrock(credentials_profile_name="default",region_name="us-east-1",model_id="amazon.titan-text-lite-v1")
def bedrockLlm(credentials_profile_name,region_name,model_id):
    bedrock_llm=Bedrock(credentials_profile_name=credentials_profile_name,region_name=region_name,model_id=model_id)
    return bedrock_llm

if __name__ == '__main__':

    query = "Who is/are the author of the document?"
    ollama_embedding = initiateOllamaEmbedding("nomic-embed-text")
    vector_data = loadVectorDB("DBIndex",ollama_embedding)
    llm = localLlama3()
    #llm = localLlama2()
    #llm = bedrockLlm("default","us-east-1","amazon.titan-text-lite-v1")
    prompt = generatePrompt()
    document_chain = create_stuff_documents_chain(llm,prompt,output_parser=JsonOutputParser())
    print(document_chain)
    print("\n------\n")
    retriever = vector_data.as_retriever()
    docs = retriever.invoke(query)
    print(docs)
    print("\n------\n")
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    print(retrieval_chain)
    print("\n------\n")
    result = retrieval_chain.invoke({"context":docs,"input":query})
    print(result['answer'])