from langchain_openrouter import ChatOpenRouter

from langchain_openai import OpenAIEmbeddings

from langchain_core.vectorstores import InMemoryVectorStore
import os

embeddings = OpenAIEmbeddings(
    model="openai/text-embedding-3-large",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    # Send raw text instead of tiktoken token IDs (OpenRouter expects strings)
    check_embedding_ctx_length=False,
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)

text = "LangChain is the framework for building context-aware reasoning applications"

vectorstore = InMemoryVectorStore.from_texts(
    [text],
    embedding=embeddings,
)

# Use the vectorstore as a retriever
retriever = vectorstore.as_retriever()

# Retrieve the most similar text
retrieved_documents = retriever.invoke("What is LangChain?")

# show the retrieved document's content
print(retrieved_documents[0].page_content)

exit()

model = ChatOpenRouter(model="openai/gpt-4o-mini", temperature=0.7, max_tokens=1000,api_key=os.getenv("OPENROUTER_API_KEY")) 
conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to German."},
    
    {"role": "user", "content": "Translate: I love building applications."}
]

#response = model.invoke(conversation)
#print(response.content) 

stream = model.stream_events("Write a short poem about the sea.", version="v3")
for token in stream.text:
    print(token, end="", flush=True)