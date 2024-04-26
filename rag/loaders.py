from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import AmazonTextractPDFLoader
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
    None
def textCsvFileLoader(file):
    None

if __name__ == "__main__":
    pdffile = sys.argv[1]
    documents = textPdfLoader(pdffile)
    print(documents[:1])
    #documents = scanDocumentLoader("scansmpl.pdf","us-east-1")
    #print(documents[:1])