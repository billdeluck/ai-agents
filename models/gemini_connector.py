# models/gemini_connector.py
import os
import logging
from typing import List, Dict, Any, Optional

# Make sure to install the library: pip install google-generativeai
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from .base_model import BaseModel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv(dotenv_path='settings/api_keys.env')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiConnector(BaseModel):
    """
    Connector for interacting with Google's Gemini models.
    Uses the google-generativeai library.
    """

    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the Google Gemini connector.

        Args:
            model_id (str): The specific Gemini model ID (e.g., "models/gemini-pro", "models/gemini-1.5-pro-latest").
                            This should match the identifiers used by the google-generativeai library.
            config (Optional[Dict[str, Any]]): Configuration specific to this connector.
        """
        # Note: The 'model_id' passed here *is* the api_model_name from the registry
        super().__init__(model_id, config)
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.error("GEMINI_API_KEY not found in environment variables.")
            raise ValueError("GEMINI_API_KEY is required for GeminiConnector.")

        try:
            # Configure the SDK
            genai.configure(api_key=self.api_key)
            # Create the model instance
            self.model = genai.GenerativeModel(self.model_id)
            logger.info(f"Successfully configured Gemini SDK for model: {self.model_id}")
        except Exception as e:
            logger.error(f"Failed to configure Gemini SDK or instantiate model {self.model_id}: {e}", exc_info=True)
            raise RuntimeError(f"Gemini SDK configuration/initialization failed: {e}") from e

    def _prepare_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Converts the standard message format to Gemini's format.
        Handles potential role issues (must alternate user/model).
        Merges consecutive messages from the same role if necessary.
        """
        gemini_messages = []
        last_role = None
        current_content = []

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if not role or not content:
                logger.warning(f"Skipping message with missing role or content: {msg}")
                continue

            # Map 'assistant' role to 'model' for Gemini
            gemini_role = "model" if role == "assistant" else "user"

            if gemini_role == last_role:
                # Merge consecutive messages of the same role
                current_content.append(content)
            else:
                # Add the previous accumulated content if any
                if last_role is not None and current_content:
                     gemini_messages.append({"role": last_role, "parts": [{"text": "\n".join(current_content)}]})

                # Start new message block
                current_content = [content]
                last_role = gemini_role

        # Add the last accumulated message block
        if last_role is not None and current_content:
             gemini_messages.append({"role": last_role, "parts": [{"text": "\n".join(current_content)}]})


        # Gemini API requires conversation to start with a 'user' role
        if gemini_messages and gemini_messages[0]["role"] == "model":
             logger.warning("Gemini conversation cannot start with 'model' role. Prepending a dummy user message.")
             # Option 1: Prepend a generic user message (might alter context slightly)
             # gemini_messages.insert(0, {"role": "user", "parts": [{"text": "Start."}]})
             # Option 2: Remove the initial model message (might lose context)
             gemini_messages.pop(0)
             # Option 3: Raise an error if strict adherence is needed
             # raise ValueError("Gemini conversation history cannot start with a 'model' role.")


        return gemini_messages

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Sends messages to the specified Gemini model and returns the response.

        Args:
            messages (List[Dict[str, str]]): List of message dictionaries (user/assistant roles).
            **kwargs: Additional arguments for the Gemini API generation config.
                      Common kwargs: temperature, max_output_tokens, top_p, top_k, stop_sequences.

        Returns:
            str: The text content of the model's response.

        Raises:
            ConnectionError: If the API request fails to connect.
            PermissionError: If authentication fails or resource exhausted.
            RuntimeError: For other API errors or unexpected issues.
        """
        logger.debug(f"Sending request to Gemini model: {self.model_id} with messages: {messages}")

        try:
            # Prepare messages for Gemini format
            gemini_history = self._prepare_messages(messages[:-1]) # History is all but the last message
            last_message = messages[-1]['content'] if messages else ""

             # Handle potential empty last message (though less likely with good agent logic)
            if not last_message and not gemini_history:
                 logger.warning("Attempting Gemini chat with empty messages list.")
                 return "" # Or raise error


            # Prepare generation configuration
            generation_config = {
                "temperature": kwargs.get("temperature"),
                "max_output_tokens": kwargs.get("max_output_tokens", 2048), # Default max output
                "top_p": kwargs.get("top_p"),
                "top_k": kwargs.get("top_k"),
                "stop_sequences": kwargs.get("stop_sequences"),
                # Remove None values as API expects concrete types or omission
                **{k: v for k, v in kwargs.items() if k in ["temperature", "max_output_tokens", "top_p", "top_k", "stop_sequences"] and v is not None}
            }
            # Filter out None values from the config
            generation_config = {k: v for k, v in generation_config.items() if v is not None}


            # Create chat session if history exists, otherwise send single prompt
            if gemini_history:
                chat_session = self.model.start_chat(history=gemini_history)
                response = chat_session.send_message(last_message, generation_config=generation_config)
            else:
                # Use generate_content for single-turn requests without history
                response = self.model.generate_content(last_message, generation_config=generation_config)


            # Extract usage info if available
            try:
                usage_metadata = response.usage_metadata
                self.last_usage = {
                    "prompt_tokens": usage_metadata.prompt_token_count,
                    "completion_tokens": usage_metadata.candidates_token_count,
                    "total_tokens": usage_metadata.total_token_count,
                }
            except (AttributeError, ValueError):
                 # Usage metadata might not always be present or populated
                 logger.debug("Usage metadata not found in Gemini response.")
                 self.last_usage = {}


            # Extract response text
            # Ensure response.text exists and is not empty
            if hasattr(response, 'text') and response.text:
                 content = response.text.strip()
                 logger.debug(f"Received response from Gemini: {content[:100]}...")
                 return content
            else:
                 # Handle cases where response might be blocked or empty
                 block_reason = getattr(response, 'prompt_feedback', {}).get('block_reason', 'Unknown')
                 finish_reason = getattr(response, 'candidates', [{}])[0].get('finish_reason', 'Unknown')
                 logger.warning(f"Gemini response was empty or blocked. Block reason: {block_reason}, Finish reason: {finish_reason}")
                 # Depending on requirements, either return empty string or raise error
                 # return ""
                 raise RuntimeError(f"Gemini response blocked or empty. Finish Reason: {finish_reason}, Block Reason: {block_reason}")



        # Specific Google API exceptions
        except google_exceptions.ResourceExhausted as e:
             logger.error(f"Gemini API request failed due to resource exhaustion (likely rate limit): {e}", exc_info=True)
             raise PermissionError(f"Gemini rate limit likely exceeded: {e}") from e
        except google_exceptions.PermissionDenied as e:
             logger.error(f"Gemini API request failed due to permission denied (check API key/permissions): {e}", exc_info=True)
             raise PermissionError(f"Gemini permission denied: {e}") from e
        except google_exceptions.InvalidArgument as e:
             logger.error(f"Gemini API request failed due to invalid argument (check payload/parameters): {e}", exc_info=True)
             # You might want to inspect 'messages' and 'generation_config' here
             raise ValueError(f"Gemini invalid argument: {e}") from e
        except google_exceptions.GoogleAPIError as e: # Catch other Google-specific API errors
             logger.error(f"A Google API error occurred during Gemini call: {e}", exc_info=True)
             raise RuntimeError(f"Google API Error: {e}") from e
        # General exceptions
        except Exception as e:
            logger.error(f"An unexpected error occurred during Gemini API call: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected Gemini error: {e}") from e

# Example Usage
if __name__ == "__main__":
    try:
        # Make sure GEMINI_API_KEY is set in settings/api_keys.env
        # Use a model name exactly as expected by the SDK and registry
        connector = GeminiConnector(model_id="gemini-pro") # Or "models/gemini-1.5-pro-latest" if available/needed

        test_messages_single = [{"role": "user", "content": "What is the significance of the 'Transformer' architecture in AI?"}]

        test_messages_multi = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."},
            {"role": "user", "content": "What is a famous landmark there?"}
        ]

        print("--- Gemini Connector Test (Single Turn) ---")
        response_single = connector.chat(test_messages_single, temperature=0.6, max_output_tokens=150)
        print(f"Model: {connector.get_model_id()}")
        print(f"Test Prompt: {test_messages_single[0]['content']}")
        print(f"Response: {response_single}")
        print(f"Usage Info: {connector.get_usage()}")
        print("-----------------------------------------")

        print("\n--- Gemini Connector Test (Multi Turn) ---")
        response_multi = connector.chat(test_messages_multi, temperature=0.5)
        print(f"Model: {connector.get_model_id()}")
        print(f"Final User Prompt: {test_messages_multi[-1]['content']}")
        print(f"Response: {response_multi}")
        print(f"Usage Info: {connector.get_usage()}")
        print("----------------------------------------")

    except ValueError as ve:
        print(f"Configuration Error: {ve}")
    except PermissionError as pe:
         print(f"Permission/Rate Limit Error: {pe}")
    except RuntimeError as re:
         print(f"Runtime Error: {re}")
    except Exception as ex:
        print(f"An error occurred during testing: {ex}")