from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema.runnable import Runnable

from rag.loaders import textPdfLoader
from rag.embeddings import initiateOllamaEmbedding
from rag.faissvector import createVectorDB
from rag.llms import localLlama3
from rag.prompts import generatePrompt

def get_chain(file,query):
    """Return a chain."""

    #query = "Who is/are the author of the document?"
    query = query
    documents = textPdfLoader(file)
    ollama_embedding = initiateOllamaEmbedding(modelname="nomic-embed-text")
    db_index = createVectorDB(documents,ollama_embedding)
    prompt = generatePrompt()
    model = localLlama3()
    document_chain = create_stuff_documents_chain(model,prompt,output_parser=JsonOutputParser())
    retriever = db_index.as_retriever()
    docs = retriever.invoke(query)
    retrieval_chain = create_retrieval_chain(retriever,document_chain)
    result = retrieval_chain.invoke({"context":docs,"input":query})
    return result['answer']