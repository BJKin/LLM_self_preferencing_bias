import os

from dotenv import load_dotenv
from google import genai
from google.genai import types as genai_types
from openai import OpenAI

from pipeline.config import MODELS

load_dotenv()

MAX_ATTEMPTS = 6


def _call_openrouter(cfg: dict, prompt: str, temperature: float) -> dict:
    """ Openrouter api call """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        max_retries=MAX_ATTEMPTS - 1,
    )

    extra_body = {}
    if cfg.get("provider"):
        extra_body["provider"] = cfg["provider"]
    if cfg.get("reasoning"):
        # remove model's thinking from response
        extra_body["reasoning"] = {**cfg["reasoning"], "exclude": True}

    kwargs = dict(
        model=cfg["id"],
        messages=[{"role": "user", "content": prompt}],
        extra_body=extra_body,
    )

    # remove temperature for reasoning models (not accepted)
    if cfg.get("reasoning", {}).get("enabled"):
        temperature = None
    else:
        kwargs["temperature"] = temperature

    resp = client.chat.completions.create(**kwargs)

    choice = resp.choices[0]
    details = getattr(resp.usage, "completion_tokens_details", None)
    thinking_tokens = getattr(details, "reasoning_tokens", None) or 0
    return {
        "text": (choice.message.content or "").strip(),
        "provider": getattr(resp, "provider", None),
        "temperature": temperature,
        "input_tokens": resp.usage.prompt_tokens,
        "output_tokens": resp.usage.completion_tokens - thinking_tokens,
        "thinking_tokens": thinking_tokens,
        "finish_reason": choice.finish_reason,
    }


def _call_gemini(cfg: dict, prompt: str) -> dict:
    """ Gemini api call """
    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"],
        http_options=genai_types.HttpOptions(
            retry_options=genai_types.HttpRetryOptions(
                attempts=MAX_ATTEMPTS, initial_delay=2, max_delay=60
            )
        ),
    )

    interaction = client.interactions.create(
        model=cfg["id"],
        input=prompt,
        generation_config={"thinking_level": cfg["thinking_level"]},
    )

    usage = interaction.usage
    return {
        "text": interaction.output_text.strip(), 
        "provider": "google",
        "temperature": None,
        "input_tokens": usage.total_input_tokens,
        "output_tokens": usage.total_output_tokens,
        "thinking_tokens": usage.total_thought_tokens,
        "finish_reason": interaction.status,
    }


def generate(model: str, prompt: str, temperature: float) -> dict:
    """ LLM api call wrapper """
    cfg = MODELS[model]
    if cfg["backend"] == "openrouter":
        result = _call_openrouter(cfg, prompt, temperature)
    elif cfg["backend"] == "gemini":
        result = _call_gemini(cfg, prompt)
    return {"model": model, "model_id": cfg["id"], **result}
