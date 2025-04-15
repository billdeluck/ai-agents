# models/model_registry.py
from typing import Dict, Any, List, Optional # <--- Add Optional here

# Define the structure for model metadata clearly
ModelMetadata = Dict[str, Any] # Type alias for clarity

"""
Central registry for available language models and their metadata.
This dictionary stores information about each model's capabilities, provider,
cost tier, limits, and potential fallbacks. The ModelRouter uses this
registry to make informed decisions about which model to use for a given task.

Structure of each entry:
{
    "model_id": { # The unique identifier used internally and potentially by providers
        "provider": "provider_name",      # e.g., "openai", "anthropic", "google"
        "connector_class": "ClassName",   # The string name of the connector class in models/
        "api_model_name": "actual_api_id",# The specific name used in the API call (e.g., gpt-4-turbo)
        "capabilities": ["list", "of", "tags"], # e.g., chat, code, vision, long_context, functions
        "cost_tier": "low|medium|high|premium", # Relative cost category
        "preferred_priority": 1,         # Lower number means higher priority for preference matching
        "limits": {                      # Optional: Known limits
            "max_tokens": 128000,
            "rate_limit_rpm": 500        # Example: Requests Per Minute
        },
        "fallbacks": ["list", "of", "model_ids"] # Ordered list of models to try if this one fails
    },
    ...
}
"""

MODEL_REGISTRY: Dict[str, ModelMetadata] = {
    # --- OpenAI Models ---
    "gpt-4-turbo": {
        "provider": "openai",
        "connector_class": "OpenAIConnector",
        "api_model_name": "gpt-4-turbo-preview", # Or specific preview/stable version
        "capabilities": ["chat", "code", "functions", "reasoning", "long_context"],
        "cost_tier": "premium",
        "preferred_priority": 1,
        "limits": {"max_tokens": 128000, "rate_limit_rpm": 500}, # Example limits
        "fallbacks": ["gpt-4", "gpt-3.5-turbo"]
    },
     "gpt-4": {
        "provider": "openai",
        "connector_class": "OpenAIConnector",
        "api_model_name": "gpt-4",
        "capabilities": ["chat", "code", "functions", "reasoning"],
        "cost_tier": "high",
        "preferred_priority": 2,
        "limits": {"max_tokens": 8192, "rate_limit_rpm": 200}, # Example limits
        "fallbacks": ["gpt-3.5-turbo"]
    },
    "gpt-3.5-turbo": {
        "provider": "openai",
        "connector_class": "OpenAIConnector",
        "api_model_name": "gpt-3.5-turbo",
        "capabilities": ["chat", "code", "functions", "fast"],
        "cost_tier": "low",
        "preferred_priority": 5,
        "limits": {"max_tokens": 16385, "rate_limit_rpm": 3500}, # Example limits (often higher)
        "fallbacks": [] # Often the last resort for OpenAI
    },

    # --- Anthropic Models ---
    "claude-3-opus": {
        "provider": "anthropic",
        "connector_class": "ClaudeConnector",
        "api_model_name": "claude-3-opus-20240229",
        "capabilities": ["chat", "code", "reasoning", "long_context", "vision", "complex_tasks"],
        "cost_tier": "premium",
        "preferred_priority": 1,
        "limits": {"max_tokens": 200000, "rate_limit_rpm": 50}, # Example limits
        "fallbacks": ["claude-3-sonnet"]
    },
    "claude-3-sonnet": {
        "provider": "anthropic",
        "connector_class": "ClaudeConnector",
        "api_model_name": "claude-3-sonnet-20240229",
        "capabilities": ["chat", "code", "reasoning", "long_context", "vision", "balanced"],
        "cost_tier": "high",
        "preferred_priority": 3,
        "limits": {"max_tokens": 200000, "rate_limit_rpm": 100}, # Example limits
        "fallbacks": ["claude-3-haiku"]
    },
    "claude-3-haiku": {
        "provider": "anthropic",
        "connector_class": "ClaudeConnector",
        "api_model_name": "claude-3-haiku-20240307",
        "capabilities": ["chat", "code", "vision", "fast", "simple_tasks"],
        "cost_tier": "low",
        "preferred_priority": 6,
        "limits": {"max_tokens": 200000, "rate_limit_rpm": 200}, # Example limits
        "fallbacks": []
    },

    # --- Google Models (Add GeminiConnector later) ---
    "gemini-1.5-pro": {
         "provider": "google",
         "connector_class": "GeminiConnector", # Assumes GeminiConnector exists
         "api_model_name": "models/gemini-1.5-pro-latest",
         "capabilities": ["chat", "code", "reasoning", "long_context", "vision", "audio", "multimodal"],
         "cost_tier": "premium", # Assuming
         "preferred_priority": 1,
         "limits": {"max_tokens": 1048576}, # Context window, rate limits vary
         "fallbacks": ["gemini-pro"]
    },
     "gemini-pro": {
         "provider": "google",
         "connector_class": "GeminiConnector", # Assumes GeminiConnector exists
         "api_model_name": "models/gemini-pro",
         "capabilities": ["chat", "code", "reasoning", "vision"], # Check specific capabilities
         "cost_tier": "medium", # Assuming
         "preferred_priority": 4,
         "limits": {"max_tokens": 30720}, # Rate limits vary
         "fallbacks": []
    },

    # --- Placeholders for other providers (TogetherAI, HuggingFace) ---
    # "llama3-8b-instruct": {
    #     "provider": "togetherai",
    #     "connector_class": "TogetherAIConnector",
    #     "api_model_name": "meta-llama/Llama-3-8b-chat-hf",
    #     "capabilities": ["chat", "code", "fast"],
    #     "cost_tier": "low",
    #     "preferred_priority": 7,
    #     "limits": {}, # Check provider limits
    #     "fallbacks": []
    # },
    # "mistral-7b-instruct": {
    #     "provider": "huggingface", # Or another provider
    #     "connector_class": "HuggingFaceConnector",
    #     "api_model_name": "mistralai/Mistral-7B-Instruct-v0.2",
    #     "capabilities": ["chat", "fast", "simple_tasks"],
    #     "cost_tier": "very_low",
    #     "preferred_priority": 8,
    #     "limits": {}, # Check provider limits
    #     "fallbacks": []
    # },
}

MODEL_REGISTRY.update({
    "deepseek-chat": { # Internal ID we choose
        "provider": "openrouter",
        "connector_class": "OpenRouterConnector",
        "api_model_name": "deepseek/deepseek-chat", # Exact ID for OpenRouter API
        "capabilities": ["chat", "code", "reasoning", "fast"], # Adjust based on DeepSeek's known strengths
        "cost_tier": "medium", # Or "low"/"free" depending on which version (check OpenRouter pricing)
        "preferred_priority": 5, # Assign a priority level
        "limits": {}, # Define if known (check OpenRouter docs for rate limits)
        "fallbacks": ["deepseek-coder"] # Example fallback
    },
    "deepseek-coder": { # Internal ID
        "provider": "openrouter",
        "connector_class": "OpenRouterConnector",
        "api_model_name": "deepseek/deepseek-coder", # Exact ID for OpenRouter API
        "capabilities": ["chat", "code", "reasoning", "coding_tasks"],
        "cost_tier": "medium", # Or "low"/"free"
        "preferred_priority": 4, # Higher priority for coding tasks maybe?
        "limits": {},
        "fallbacks": ["gpt-3.5-turbo"] # Example fallback to another provider
    },
    # Example for the free tier if you want to explicitly target it
    "deepseek-chat-free": {
         "provider": "openrouter",
         "connector_class": "OpenRouterConnector",
         "api_model_name": "deepseek/deepseek-chat", # Map to the same base model for now
         # OR if OpenRouter has a specific :free identifier use that:
         # "api_model_name": "deepseek/deepseek-chat:free", # As per user instructions - use this if valid!
         "capabilities": ["chat", "code", "reasoning", "fast"],
         "cost_tier": "free", # Explicitly mark as free
         "preferred_priority": 8, # Lower priority than paid usually
         "limits": {"rate_limit_rpm": 10}, # Example: Put known free tier limits here
         "fallbacks": []
     },
    # Add other OpenRouter models as needed (e.g., Mistral, Llama)
    # "mistral-7b-instruct-openrouter": {
    #     "provider": "openrouter",
    #     "connector_class": "OpenRouterConnector",
    #     "api_model_name": "mistralai/mistral-7b-instruct",
    #     "capabilities": ["chat", "fast", "simple_tasks"],
    #     "cost_tier": "very_low",
    #     "preferred_priority": 9,
    #     "limits": {},
    #     "fallbacks": []
    # },
})

def get_model_metadata(model_id: str) -> Optional[ModelMetadata]: # <- Optional is used here
    """Helper function to safely retrieve metadata for a given model ID."""
    return MODEL_REGISTRY.get(model_id)

def get_model_ids_by_capability(capability: str) -> List[str]:
    """Helper function to find model IDs that have a specific capability."""
    return [
        model_id for model_id, meta in MODEL_REGISTRY.items()
        if capability in meta.get("capabilities", [])
    ]

def get_all_model_ids() -> List[str]:
    """Returns a list of all registered model IDs."""
    return list(MODEL_REGISTRY.keys())

# Example Usage
if __name__ == "__main__":
    print("--- Model Registry Examples ---")
    print(f"All registered models: {get_all_model_ids()}")
    gpt4_meta = get_model_metadata("gpt-4-turbo")
    if gpt4_meta:
        print(f"\nMetadata for gpt-4-turbo:")
        for key, value in gpt4_meta.items():
            print(f"  {key}: {value}")

    code_models = get_model_ids_by_capability("code")
    print(f"\nModels with 'code' capability: {code_models}")

    vision_models = get_model_ids_by_capability("vision")
    print(f"Models with 'vision' capability: {vision_models}")
    print("-----------------------------")


    print(f"\nMetadata for deepseek-chat:")
    ds_meta = get_model_metadata("deepseek-chat")
    if ds_meta:
         for key, value in ds_meta.items():
             print(f"  {key}: {value}")
    else:
        print("  deepseek-chat not found in registry.")