"""
Shared LLM client factory with structured output and retry logic.
"""
import asyncio
import logging
from typing import TypeVar, Type

from anthropic import RateLimitError
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from app.config import settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

MAX_RETRIES = 8
INITIAL_DELAY = 5.0  # seconds


def get_llm(max_tokens: int = 8192) -> ChatAnthropic:
    return ChatAnthropic(
        model=settings.anthropic_model,
        temperature=0,
        max_tokens=max_tokens,
        api_key=settings.anthropic_api_key,
    )


async def invoke_with_retry(llm, messages: list[BaseMessage]):
    """Invoke LLM with exponential backoff on rate limit errors."""
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await llm.ainvoke(messages)
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "Rate limited (attempt %d/%d), waiting %.0fs: %s",
                attempt + 1, MAX_RETRIES, delay, e,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)


async def invoke_structured(
    model: Type[T],
    messages: list[BaseMessage],
    agent_name: str,
) -> T:
    """Invoke LLM with structured output enforced via tool calling."""
    llm = get_llm()
    structured_llm = llm.with_structured_output(model)
    delay = INITIAL_DELAY
    for attempt in range(MAX_RETRIES + 1):
        try:
            return await structured_llm.ainvoke(messages)
        except RateLimitError as e:
            if attempt == MAX_RETRIES:
                raise
            logger.warning(
                "%s rate limited (attempt %d/%d), waiting %.0fs",
                agent_name, attempt + 1, MAX_RETRIES, delay,
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60.0)
