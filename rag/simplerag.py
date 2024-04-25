from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.llms import Bedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#loader = TextLoader("test.txt")
loader = PyPDFLoader("research.pdf")
#print(loader)
text_documents = loader.load()
#print(text_documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#text_splitter1 = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents = text_splitter.split_documents(text_documents)
#documents1 = text_splitter1.split_documents(text_documents)
#print(documents[:1])
#print(documents1[:1])
embeddings = OllamaEmbeddings(model="nomic-embed-text")
#embeddings = OllamaEmbeddings(model="llama3")
#bedrockembeddings = BedrockEmbeddings(credentials_profile_name="default",region_name="us-east-1",model_id="amazon.titan-embed-text-v1")
#embedded_data = embeddings.embed_query(documents)
#print(embedded_data[:10])
#vector_db = Chroma.from_documents(documents[:5],bedrockembeddings)
vector_db = FAISS.from_documents(documents[:5],embeddings)
#print(type(vector_db))
#query = "Who is the Author"
#result = vector_db.similarity_search(query)
#print(result[0].page_content)

#llm=Ollama(model="Llama2")
llm=Ollama(model="Llama3",stop=['<|eot_id|>'])
#bedrockllm=Bedrock(credentials_profile_name="default",region_name="us-east-1",model_id="amazon.titan-text-lite-v1")
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context.
                                          Think step by step before providing the detailed answer.
                                          Provide answer in a json format.
                                          <context>
                                          {context}
                                          </context>
                                          Question: {input}
                                          """)

document_chain = create_stuff_documents_chain(llm,prompt)
retriever = vector_db.as_retriever()
#print(retriever)
retrieval_chain = create_retrieval_chain(retriever,document_chain)
result = retrieval_chain.invoke({"input":"What this document is about"})
print(result['answer'])