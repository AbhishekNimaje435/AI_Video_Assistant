import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.3,
    )


def split_transcript(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def summarize(transcript: str) -> str:
    llm = get_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Summarize this portion of the meeting transcript in concise bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript(transcript)

    partial_summaries = []

    for chunk in chunks:
        summary = map_chain.invoke({"text": chunk})
        partial_summaries.append(summary)

    combined_text = "\n\n".join(partial_summaries)

    final_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Combine the following partial meeting summaries into one clear, professional summary with bullet points.",
            ),
            ("human", "{text}"),
        ]
    )

    final_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | final_prompt
        | llm
        | StrOutputParser()
    )

    return final_chain.invoke(combined_text)


def generate_title(transcript: str) -> str:
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Generate a short professional meeting title (maximum 8 words). Return only the title.",
            ),
            ("human", "{text}"),
        ]
    )

    chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain.invoke(transcript[:2000]).strip()