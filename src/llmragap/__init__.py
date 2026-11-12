import os

from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _require_openai_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Export it in your shell before running llmragap."
        )
    return api_key


def run_app(question: str = "What is Chroma?") -> str:
    _require_openai_key()

    raw_documents = [
        "LangChain is a framework for developing applications powered by language models.",
        "Chroma is an open-source vector database designed to make it easy to build AI applications with embeddings.",
        "Retrieval-Augmented Generation (RAG) is a technique for enhancing LLM responses with external data.",
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents(raw_documents)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(docs, embeddings, collection_name="llmragap-docs")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    prompt = ChatPromptTemplate.from_template(
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    rag_chain = (
        RunnablePassthrough.assign(context=retriever)
        | prompt
        | llm
        | StrOutputParser()
    )

    answer = rag_chain.invoke(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    return answer


def main() -> None:
    print("Hello from llmragap!")
    try:
        run_app()
    except RuntimeError as exc:
        print(exc)


if __name__ == "__main__":
    main()
