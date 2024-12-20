class LLMConfig:
    OLLAMA = "ollama"
    OPENAI = "openai"

    PROVIDERS = {
        OLLAMA: {
            "available_models": ["llama3.3", "llama3.2", "llama3.1"],
            "default_model": "llama3.3",
        },
        OPENAI: {
            "available_models": ["gpt-4o-mini", "o1-mini-2024-09-12"],
            "default_model": "gpt-4o-mini",
            "api_key_required": True
        }
    }
    
    @staticmethod
    def validate_model(provider: str, model_name: str) -> bool:
        if provider not in LLMConfig.PROVIDERS:
            return False
        return model_name in LLMConfig.PROVIDERS[provider]["available_models"]