import os

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough


def get_llm():
    return ChatMistralAI(
        model="mistral-small-latest",
        mistral_api_key=os.getenv("MISTRAL_API_KEY"),
        temperature=0.2,
    )


def build_chain(system_prompt: str):
    llm = get_llm()

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
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

    return chain


def extract_action_items(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all action items.

For each action item provide:
- Task
- Owner
- Deadline (or "Not specified")

Return as a numbered list.

If nothing is found, return:
No action items found.
"""
    )

    return chain.invoke(transcript)


def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all key decisions made during the meeting.

Return as a numbered list.

If nothing is found, return:
No key decisions found.
"""
    )

    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """
You are an expert meeting analyst.

Extract all unresolved questions or follow-up items.

Return as a numbered list.

If nothing is found, return:
No open questions found.
"""
    )

    return chain.invoke(transcript)