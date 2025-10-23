from langchain_openai import ChatOpenAI
from ..core.config import get_settings
settings=get_settings()
mcqa_chat_llm = ChatOpenAI(
    openai_api_key=settings.api_key,
    openai_api_base=settings.api_base,
    model="my_lora",
    temperature=0,
    max_tokens=200
)

