from langchain_groq import ChatGroq
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv()


@lru_cache(maxsize=1)
def get_llm_for_tools():
    return ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.1,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_report_writer_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.0,
    )

@lru_cache(maxsize=1)
def get_synthesis_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,      
        max_tokens=4096,      
    )

@lru_cache(maxsize=1)
def get_chatbot_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,      
        max_tokens=4096,      
    )