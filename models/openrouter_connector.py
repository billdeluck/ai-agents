# models/openrouter_connector.py
import os
import logging
import requests # Using requests library
from typing import List, Dict, Any, Optional

from .base_model import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='settings/api_keys.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenRouterConnector(BaseModel):
    """
    Connector for interacting with models hosted on OpenRouter.ai.
    Uses the OpenAI-compatible chat completions endpoint.
    Allows access to various models like DeepSeek, Mistral, etc.
    """
    API_URL = "https://openrouter.ai/api/v1/chat/completions"

    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the OpenRouter connector.

        Args:
            model_id (str): The specific OpenRouter model identifier
                            (e.g., "deepseek/deepseek-chat:free", "mistralai/mistral-7b-instruct").
                            This is the 'api_model_name' from the registry.
            config (Optional[Dict[str, Any]]): Configuration specific to this connector.
        """
        # The model_id passed here *is* the api_model_name expected by OpenRouter
        super().__init__(model_id, config)
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            logger.error("OPENROUTER_API_KEY not found in environment variables.")
            raise ValueError("OPENROUTER_API_KEY is required for OpenRouterConnector.")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # Optional: Add HTTP Referer or other headers if needed/recommended by OpenRouter
            # "HTTP-Referer": "YOUR_SITE_URL", # Example
            # "X-Title": "YOUR_APP_NAME",     # Example
        }

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Sends messages to the specified model via OpenRouter.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries (user/assistant roles).
            **kwargs: Additional arguments compatible with OpenAI's API structure
                      (e.g., temperature, max_tokens, top_p, stream).

        Returns:
            str: The content of the assistant's response.

        Raises:
            ConnectionError: If the API request fails to connect.
            PermissionError: If authentication fails or rate limits are hit.
            RuntimeError: For other API errors or unexpected issues.
        """
        logger.debug(f"Sending request to OpenRouter model: {self.model_id} with messages: {messages}")

        # Prepare payload - mirrors OpenAI structure
        payload = {
            "model": self.model_id, # Use the specific model ID passed during init
            "messages": messages,
            # Include common parameters if present in kwargs
            **{k: v for k, v in kwargs.items() if k in [
                "temperature", "max_tokens", "top_p", "frequency_penalty",
                "presence_penalty", "stop", "stream" # Add others if needed
               ] and v is not None}
        }

        try:
            response = requests.post(self.API_URL, headers=self.headers, json=payload, timeout=120) # Add timeout
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()

            # Extract usage info (assuming OpenAI-compatible format)
            if 'usage' in response_data:
                self.last_usage = {
                    "prompt_tokens": response_data['usage'].get('prompt_tokens', 0),
                    "completion_tokens": response_data['usage'].get('completion_tokens', 0),
                    "total_tokens": response_data['usage'].get('total_tokens', 0),
                }
            else:
                self.last_usage = {}

            # Extract content (assuming OpenAI-compatible format)
            if response_data.get("choices") and isinstance(response_data["choices"], list) and len(response_data["choices"]) > 0:
                choice = response_data["choices"][0]
                if choice.get("message") and choice["message"].get("content"):
                    content = choice["message"]["content"].strip()
                    logger.debug(f"Received response from OpenRouter ({self.model_id}): {content[:100]}...")
                    return content
                else:
                    finish_reason = choice.get('finish_reason', 'unknown')
                    logger.error(f"OpenRouter response choice missing message/content. Finish reason: {finish_reason}. Data: {response_data}")
                    raise RuntimeError(f"OpenRouter response format error: Missing message content (Finish reason: {finish_reason}).")
            else:
                logger.error(f"Unexpected response format from OpenRouter: No valid choices found. Data: {response_data}")
                raise RuntimeError("OpenRouter response did not contain expected choices format.")

        except requests.exceptions.ConnectionError as e:
            logger.error(f"OpenRouter API request failed to connect: {e}", exc_info=True)
            raise ConnectionError(f"OpenRouter API connection error: {e}") from e
        except requests.exceptions.Timeout as e:
             logger.error(f"OpenRouter API request timed out: {e}", exc_info=True)
             raise TimeoutError(f"OpenRouter API request timed out: {e}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_details = e.response.text
            logger.error(f"OpenRouter API returned an error status: {status_code} - {error_details}", exc_info=True)
            if status_code == 401 or status_code == 403:
                raise PermissionError(f"OpenRouter authentication/permission error ({status_code}): {error_details}") from e
            elif status_code == 429:
                 # OpenRouter might provide specific rate limit info in headers or body
                 raise PermissionError(f"OpenRouter rate limit exceeded ({status_code}): {error_details}") from e
            else:
                raise RuntimeError(f"OpenRouter API error ({status_code}): {error_details}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenRouter API call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected OpenRouter error: {e}") from e

# Example Usage
if __name__ == "__main__":
    try:
        # Make sure OPENROUTER_API_KEY is set
        # Use a specific DeepSeek model identifier from OpenRouter
        connector = OpenRouterConnector(model_id="deepseek/deepseek-chat") # Or :free if testing free tier
        test_messages = [{"role": "user", "content": "Explain the concept of 'emergence' in complex systems."}]
        response_content = connector.chat(test_messages, temperature=0.7, max_tokens=250)
        print("--- OpenRouter Connector Test (DeepSeek) ---")
        print(f"Model: {connector.get_model_id()}")
        print(f"Test Prompt: {test_messages[0]['content']}")
        print(f"Response: {response_content}")
        print(f"Usage Info: {connector.get_usage()}")
        print("------------------------------------------")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except PermissionError as pe:
         print(f"Permission/Rate Limit Error: {pe}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")