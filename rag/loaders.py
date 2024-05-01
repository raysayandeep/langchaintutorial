from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import boto3
import sys

def scanDocumentLoader(file,textract_region_name):
    textract_client = boto3.client("textract", region_name=textract_region_name)
    loader = AmazonTextractPDFLoader(file,client=textract_client)
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    documents = text_splitter.split_documents(text_documents)
    return documents

def textPdfLoader(file):
    loader = PDFMinerLoader(file)
    text_documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    documents = text_splitter.split_documents(text_documents)
    return documents

def excelFileLoader(file):
    loader = UnstructuredExcelLoader(file,mode="elements")
    text_documents = loader.load()
    content = text_documents[0].page_content
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
    documents = text_splitter.split_documents(content)
    return documents

def textCsvFileLoader(file):
    None

if __name__ == "__main__":
    file = sys.argv[1]
    documents = excelFileLoader(file)
    print(documents)
    #documents = scanDocumentLoader("scansmpl.pdf","us-east-1")
    #print(documents[:1])