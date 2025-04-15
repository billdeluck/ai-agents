# models/claude_connector.py
import os
import logging
import requests # Using requests library for Anthropic API
from typing import List, Dict, Any

from .base_model import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='settings/api_keys.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ClaudeConnector(BaseModel):
    """
    Connector for interacting with Anthropic's Claude models.
    Uses the Anthropic Messages API.
    """
    API_URL = "https://api.anthropic.com/v1/messages"
    # Consider making version configurable if needed
    ANTHROPIC_VERSION = "2023-06-01"

    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        """
        Initializes the Anthropic Claude connector.

        Args:
            model_id (str): The specific Anthropic model ID (e.g., "claude-3-opus-20240229").
            config (Dict[str, Any]): Configuration specific to this connector.
        """
        super().__init__(model_id, config)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.error("ANTHROPIC_API_KEY not found in environment variables.")
            raise ValueError("ANTHROPIC_API_KEY is required for ClaudeConnector.")

        self.headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Sends messages to the specified Claude model using the Messages API.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries. The API expects alternating user/assistant roles.
                                             A system prompt can be passed via kwargs['system'].
            **kwargs: Additional arguments for the Anthropic API.
                      Common kwargs: system (str), max_tokens (int), temperature (float), top_p (float), top_k (int).

        Returns:
            str: The content of the assistant's response.

        Raises:
            ConnectionError: If the API request fails to connect.
            PermissionError: If authentication fails or rate limits are hit.
            RuntimeError: For other API errors or unexpected issues.
        """
        logger.debug(f"Sending request to Anthropic model: {self.model_id} with messages: {messages}")

        # Prepare payload according to Anthropic Messages API spec
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", 1024), # Anthropic requires max_tokens
            # Add other parameters if provided in kwargs
            **{k: v for k, v in kwargs.items() if k in ["system", "temperature", "top_p", "top_k", "stop_sequences"]}
        }

        try:
            response = requests.post(self.API_URL, headers=self.headers, json=payload)
            response.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)

            response_data = response.json()

            # Extract usage info
            if 'usage' in response_data:
                 self.last_usage = {
                     "prompt_tokens": response_data['usage'].get('input_tokens', 0),
                     "completion_tokens": response_data['usage'].get('output_tokens', 0),
                     "total_tokens": response_data['usage'].get('input_tokens', 0) + response_data['usage'].get('output_tokens', 0)
                 }
            else:
                 self.last_usage = {}


            # Extract content - expects a list of content blocks, usually one text block
            if response_data.get("content") and isinstance(response_data["content"], list):
                # Find the first text block
                text_content = next((block.get("text", "") for block in response_data["content"] if block.get("type") == "text"), "")
                content = text_content.strip()
                logger.debug(f"Received response from Anthropic: {content[:100]}...")
                return content
            else:
                 logger.error(f"Unexpected response format from Anthropic: {response_data}")
                 raise RuntimeError("Anthropic response did not contain expected content format.")


        except requests.exceptions.ConnectionError as e:
            logger.error(f"Anthropic API request failed to connect: {e}", exc_info=True)
            raise ConnectionError(f"Anthropic API connection error: {e}") from e
        except requests.exceptions.Timeout as e:
             logger.error(f"Anthropic API request timed out: {e}", exc_info=True)
             raise TimeoutError(f"Anthropic API request timed out: {e}") from e
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            error_details = e.response.text
            logger.error(f"Anthropic API returned an error status: {status_code} - {error_details}", exc_info=True)
            if status_code == 401 or status_code == 403:
                raise PermissionError(f"Anthropic authentication/permission error ({status_code}): {error_details}") from e
            elif status_code == 429:
                 raise PermissionError(f"Anthropic rate limit exceeded ({status_code}): {error_details}") from e
            else:
                raise RuntimeError(f"Anthropic API error ({status_code}): {error_details}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during Anthropic API call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Anthropic error: {e}") from e


# Example Usage
if __name__ == "__main__":
    try:
        # Make sure ANTHROPIC_API_KEY is set
        # Use a known valid model ID
        connector = ClaudeConnector(model_id="claude-3-haiku-20240307") # Haiku is often fastest/cheapest for testing
        test_messages = [{"role": "user", "content": "What is the core idea behind the 'Chain of Thought' prompting technique?"}]
        response_content = connector.chat(test_messages, temperature=0.5, max_tokens=200)
        print("--- Claude Connector Test ---")
        print(f"Model: {connector.get_model_id()}")
        print(f"Test Prompt: {test_messages[0]['content']}")
        print(f"Response: {response_content}")
        print(f"Usage Info: {connector.get_usage()}")
        print("---------------------------")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")