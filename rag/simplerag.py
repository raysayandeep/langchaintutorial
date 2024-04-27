from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_community.llms import Bedrock
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import boto3
import os
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
#loader = TextLoader("test.txt")
#loader = PyPDFLoader("research.pdf")
loader = PDFMinerLoader("research1.pdf")
#textract_client = boto3.client("textract", region_name="us-east-1")
#loader = AmazonTextractPDFLoader("scansmpl.pdf",client=textract_client)
#print(loader)
text_documents = loader.load()
#print(text_documents)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
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
query = "Who is/are the author of the document?"
#query = "What this document explains about?"
#result = vector_db.similarity_search(query)
#print(result[0].page_content)

#llm=Ollama(model="Llama2")
llm=Ollama(model="Llama3",stop=['<|eot_id|>'])
#bedrockllm=Bedrock(credentials_profile_name="default",region_name="us-east-1",model_id="amazon.titan-text-lite-v1")
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context.
                                          Think step by step before providing the detailed answer.
                                          Provide the answer as a JSON with key and value pairs and no premable or explaination.
                                          <context>
                                          {context}
                                          </context>
                                          question: {input}
                                          """)

prompt1 = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

document_chain = create_stuff_documents_chain(llm,prompt,output_parser=JsonOutputParser())
retriever = vector_db.as_retriever()
#docs = retriever.invoke(query)
#print(docs)
#print(retriever)
retrieval_chain = create_retrieval_chain(retriever,document_chain)
#retrieval_chain = prompt | llm | JsonOutputParser()
#result = retrieval_chain.invoke({"input":query,"context":docs})
result = retrieval_chain.invoke({"input":query})
print(result['answer'])