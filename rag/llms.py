from langchain_community.llms import Ollama
from langchain_community.llms import Bedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser

from faissvector import loadVectorDB
from embeddings import initiateOllamaEmbedding

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

    prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context.
                                          Think step by step before providing the detailed answer.
                                          Provide the answer as a JSON with key and value pairs and no premable or explaination.
                                          <context>
                                          {context}
                                          </context>
                                          question: {input}
                                          """)
    
    document_chain = create_stuff_documents_chain(llm,prompt,output_parser=JsonOutputParser())
    retriever = vector_data.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    result = retrieval_chain.invoke({"input":query})
    print(result['answer']) 
