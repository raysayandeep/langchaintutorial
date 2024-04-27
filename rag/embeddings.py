from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import BedrockEmbeddings

#embeddings = OllamaEmbeddings(model="nomic-embed-text")
#embeddings = OllamaEmbeddings(model="llama3")
#bedrockembeddings = BedrockEmbeddings(credentials_profile_name="default",region_name="us-east-1",model_id="amazon.titan-embed-text-v1")

#initilize Ollama embedding models. Example: model="nomic-embed-text"/model="llama3"
def initiateOllamaEmbedding(modelname):
    embeddings = OllamaEmbeddings(model=modelname)
    return embeddings

#initialize Bedrock embedding models. Example: credentials_profile_name="default",region_name="us-east-1",model_id="amazon.titan-embed-text-v1"
def initiateBedrockEmbedding(modelid,aws_credentials_profile_name,region):
    bedrockembeddings = BedrockEmbeddings(credentials_profile_name=aws_credentials_profile_name,region_name=region,model_id=modelid)
    return bedrockembeddings


