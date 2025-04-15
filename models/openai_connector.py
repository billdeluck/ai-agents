# models/openai_connector.py
import os
import logging
from typing import List, Dict, Any
import openai # Use 'openai' library version >= 1.0
# If using older version (0.x.x), the API calls are different. This assumes >= 1.0.

from .base_model import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='settings/api_keys.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIConnector(BaseModel):
    """
    Connector for interacting with OpenAI's language models (GPT-3.5, GPT-4, etc.).
    Assumes openai library version >= 1.0.
    """

    def __init__(self, model_id: str, config: Dict[str, Any] = None):
        """
        Initializes the OpenAI connector.

        Args:
            model_id (str): The specific OpenAI model ID (e.g., "gpt-4-turbo-preview", "gpt-3.5-turbo").
            config (Dict[str, Any]): Configuration specific to this connector (can be empty).
        """
        super().__init__(model_id, config)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.error("OPENAI_API_KEY not found in environment variables.")
            raise ValueError("OPENAI_API_KEY is required for OpenAIConnector.")

        # Initialize the OpenAI client
        try:
            # self.client = openai.OpenAI(api_key=self.api_key) # Standard initialization
             self.client = openai.OpenAI(api_key=self.api_key) # Standard initialization
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            raise

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Sends messages to the specified OpenAI model and returns the response.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries.
            **kwargs: Additional arguments for the OpenAI API (e.g., temperature, max_tokens).
                      Common kwargs: temperature, max_tokens, top_p, frequency_penalty, presence_penalty.

        Returns:
            str: The content of the assistant's response.

        Raises:
            Exception: If the API call fails.
        """
        logger.debug(f"Sending request to OpenAI model: {self.model_id} with messages: {messages}")
        try:
            # Default parameters - can be overridden by kwargs
            params = {
                "model": self.model_id,
                "messages": messages,
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 1500), # Adjust default as needed
                **{k: v for k, v in kwargs.items() if k not in ["temperature", "max_tokens"]} # Pass other kwargs
            }

            response = self.client.chat.completions.create(**params)

            # Store usage information
            if response.usage:
                self.last_usage = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            else:
                 self.last_usage = {}


            content = response.choices[0].message.content.strip()
            logger.debug(f"Received response from OpenAI: {content[:100]}...") # Log snippet
            return content

        except openai.APIConnectionError as e:
            logger.error(f"OpenAI API request failed to connect: {e}", exc_info=True)
            raise ConnectionError(f"OpenAI API connection error: {e}") from e
        except openai.RateLimitError as e:
            logger.error(f"OpenAI API request exceeded rate limit: {e}", exc_info=True)
            # Consider adding retry logic here or in the ModelRouter
            raise PermissionError(f"OpenAI rate limit exceeded: {e}") from e
        except openai.AuthenticationError as e:
             logger.error(f"OpenAI API authentication failed: {e}", exc_info=True)
             raise PermissionError(f"OpenAI authentication error: {e}") from e
        except openai.APIStatusError as e:
            logger.error(f"OpenAI API returned an error status: {e.status_code} - {e.response}", exc_info=True)
            raise RuntimeError(f"OpenAI API error ({e.status_code}): {e.message}") from e
        except Exception as e:
            logger.error(f"An unexpected error occurred during OpenAI API call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected OpenAI error: {e}") from e

# Example Usage (for testing the connector directly)
if __name__ == "__main__":
    try:
        # Make sure OPENAI_API_KEY is set in your .env file or environment
        connector = OpenAIConnector(model_id="gpt-3.5-turbo") # Use a readily available model
        test_messages = [{"role": "user", "content": "Explain the concept of modularity in software engineering in one sentence."}]
        response_content = connector.chat(test_messages, temperature=0.5)
        print("--- OpenAI Connector Test ---")
        print(f"Model: {connector.get_model_id()}")
        print(f"Test Prompt: {test_messages[0]['content']}")
        print(f"Response: {response_content}")
        print(f"Usage Info: {connector.get_usage()}")
        print("-----------------------------")
    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")