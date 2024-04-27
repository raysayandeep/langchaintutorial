from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

def generatePrompt():
    jsontemplate = """
    Answer the following question based only on the provided context.
    Think step by step before providing the detailed answer.
    Provide the answer as a JSON with key and value pairs and no premable or explaination.
    {context}
    {input}
    """
    prompt=ChatPromptTemplate.from_template(jsontemplate)
    return prompt


if __name__ == '__main__':
    print(generatePrompt())

