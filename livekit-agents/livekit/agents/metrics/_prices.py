"""List-price cost estimates for LLM usage, from the optional ``llm-prices`` table.

Installed with ``pip install "livekit-agents[prices]"``. Without it every estimate is ``None``.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

try:
    import llm_prices
except ImportError:  # the `prices` extra is not installed
    llm_prices = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from .usage import Cost

# `LLM.provider` of the OpenAI and Anthropic plugins is the host of their base URL. Other
# plugins report a name llm-prices already accepts ("Gemini", "Vertex AI", "AWS Bedrock",
# "MistralAI"). LiveKit Inference reports "livekit", which is never priced here.
_HOSTS = {
    "api.openai.com": "openai",
    "api.anthropic.com": "anthropic",
    "api.deepseek.com": "deepseek",
    "api.groq.com": "groq",
    "api.x.ai": "xai",
    "api.together.xyz": "together",
    "api.fireworks.ai": "fireworks",
    "api.mistral.ai": "mistral",
    "api.perplexity.ai": "perplexity",
    "openrouter.ai": "openrouter",
}


@functools.lru_cache(maxsize=128)
def _lookup(provider: str, model: str) -> llm_prices.Price | None:
    if llm_prices is None or not provider or not model:
        return None
    name = provider.strip().lower()
    if name.endswith(".openai.azure.com"):
        name = "azure"
    return llm_prices.lookup_provider(_HOSTS.get(name, provider), model)


def estimate_llm_cost(
    provider: str,
    model: str,
    *,
    input_tokens: int,
    input_cached_tokens: int,
    input_cache_creation_tokens: int,
    output_tokens: int,
) -> Cost | None:
    """Cost of one LLM request at the provider's public list price.

    ``input_tokens`` is the whole prompt, cache reads and writes included, as reported in
    ``LLMMetrics.prompt_tokens``. Cache writes are priced at the 5-minute rate. Returns ``None``
    when ``llm-prices`` is not installed, the provider is LiveKit Inference or unknown, or the
    table has no price for the model.
    """
    price = _lookup(provider, model)
    if price is None:
        return None
    cost = price.cost_usage(
        llm_prices.Usage(
            input=max(input_tokens - input_cached_tokens - input_cache_creation_tokens, 0),
            cache_read=input_cached_tokens,
            cache_write_5m=input_cache_creation_tokens,
            output=output_tokens,
        )
    )
    if cost is None:
        return None

    from .usage import Cost

    return Cost(amount=cost.total, rate_source="llm-prices", as_of=llm_prices.GENERATED)
