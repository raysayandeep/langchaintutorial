from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

def generatePrompt():
    jsontemplate = """
    You are a research paer analyzer.
    Answer the following question based only on the provided context.
    Think step by step before providing the detailed answer.
    Provide you answer with proper explanation.
    <context>
    {context}
    </context>
    Question:{input}
    """
    prompt=ChatPromptTemplate.from_template(jsontemplate)
    return prompt


if __name__ == '__main__':
    print(generatePrompt())

