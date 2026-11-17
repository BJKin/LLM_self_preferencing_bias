from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"

# Generation settings
TEMPERATURE = 1.0

# Models (llama-3.3-70b is control from original paper, all other models are new)
MODELS = {
    "llama-3.3-70b": {
        "backend": "openrouter",
        "id": "meta-llama/llama-3.3-70b-instruct",
        "provider": {
            "only": [
                "deepinfra/turbo"
            ],
            "allow_fallbacks": False
            }
    },
    "gemini-4-hydrogen": {
        "backend": "gemini",
        "id": "gemini-4-hydrogen",
        "thinking_level": "high",
        "provider": None,
    },
    "gpt-6-luna": {
        "backend": "openrouter",
        "id": "openai/gpt-6-luna",
        "reasoning": {"enabled": True},
        # OpenRouter's new account limit for Luna :(
        "rpm": 20,
        "provider": {
            "only": [
                "amazon-bedrock/global"
            ],
            "allow_fallbacks": False
            }
    },
    "claude-haiku-5.5": {
        "backend": "openrouter",
        "id": "anthropic/claude-haiku-5.5",
        "reasoning": {"enabled": True},
        "provider": {
            "only": [
                "claude-on-aws"
            ],
            "allow_fallbacks": False
            }
    },
    "deepseek-v4.1-flash": {
        "backend": "openrouter",
        "id": "deepseek/deepseek-v4.1-flash-1017",
        "reasoning": {"enabled": True},
        "provider": {
            "only": [
                "stepfun/fp8"
            ],
            "allow_fallbacks": False
            }
    }
}
