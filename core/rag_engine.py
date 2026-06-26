import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
)


# =====================================================
# LLM
# =====================================================

def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


# =====================================================
# Format Retrieved Documents
# =====================================================

def format_docs(docs):
    if not docs:
        return "No transcript context available."

    return "\n\n".join(doc.page_content for doc in docs)


# =====================================================
# Build RAG Chain
# =====================================================

def build_rag_chain(transcript: str):

    print("Building RAG Chain...")

    vector_store = build_vector_store(transcript)

    retriever = get_retriever(
        vector_store,
        k=8,
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI Video and Meeting Assistant.

You MUST answer ONLY using the transcript context.

Instructions:

- Combine information from ALL retrieved context.
- Do NOT say information is missing unless it truly does not exist.
- If the transcript contains the answer in different chunks,
  combine them into one response.
- For overview, summary or "what is this video about",
  generate a concise explanation from the retrieved context.
- If information is genuinely absent, reply exactly:

I could not find this information in the meeting transcript.

Transcript:

{context}
                """,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =====================================================
# Load Existing DB
# =====================================================

def load_rag_chain():

    vector_store = load_vector_store()

    retriever = get_retriever(
        vector_store,
        k=8,
    )

    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
You are an expert AI Video and Meeting Assistant.

Answer ONLY from the transcript.

If unavailable reply exactly:

I could not find this information in the meeting transcript.

Transcript:

{context}
                """,
            ),
            (
                "human",
                "{question}",
            ),
        ]
    )

    rag_chain = (
        {
            "context": retriever | RunnableLambda(format_docs),
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# =====================================================
# Ask Question
# =====================================================

def ask_question(rag_chain, question: str):
    return rag_chain.invoke(question)