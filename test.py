from langchain_community.llms import Ollama
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler 
                                 
llm = Ollama(model="llama3",stop=['<|eot_id|>'], 
             callback_manager = CallbackManager([StreamingStdOutCallbackHandler()]),
             temperature = 0.9,
             top_p = 1
             )

llm.invoke("Write me a essay about Importance of Water in 100 words. Display word count at the end.")