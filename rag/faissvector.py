from langchain_community.vectorstores import FAISS

from rag.loaders import textPdfLoader, scanDocumentLoader, excelFileLoader
from rag.embeddings import initiateBedrockEmbedding, initiateOllamaEmbedding

import sys
import boto3

def createVectorDB(documents,embedding):
    vector_db_index = FAISS.from_documents(documents[:5],embedding)
    return vector_db_index

def saveVectorDB(vector_db_index,savepath):
    db = vector_db_index.save_local(savepath)
    return db

def loadVectorDB(indexstore,embedding):
    vector_db_index = FAISS.load_local(indexstore,embedding,allow_dangerous_deserialization="True")
    return vector_db_index 

def saveVectorDBIncremental(vector_db_index,savepath,indexstore,embedding):
    vector_db_index_incremental = loadVectorDB(indexstore,embedding)
    vector_db_index_incremental.merge_from(vector_db_index)
    vector_db_index_incremental.save_local(savepath)

if __name__ == '__main__':
    #load document
    pdffile = sys.argv[1]
    documents = textPdfLoader(pdffile)
    print(documents[:1])
    #create embeddings
    ollama_embedding = initiateOllamaEmbedding(modelname="nomic-embed-text")
    #create DB index
    db_index = createVectorDB(documents,ollama_embedding)
    print(db_index)
    #save Vector DB
    saveVectorDB(db_index,"DBIndex")