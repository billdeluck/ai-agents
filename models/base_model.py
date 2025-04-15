# models/base_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseModel(ABC):
    """
    Abstract base class for all language model connectors.
    Defines the common interface for interacting with different LLM providers.
    """

    def __init__(self, model_id: str, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the base model.

        Args:
            model_id (str): A unique identifier for the specific model instance (e.g., "gpt-4-turbo").
            config (Optional[Dict[str, Any]]): Configuration parameters specific to the model or connector.
        """
        self.model_id = model_id
        self.config = config or {}
        self.last_usage = {} # Store token usage info if available

    @abstractmethod
    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Sends a list of messages to the language model and returns the response content.

        Args:
            messages (List[Dict[str, str]]): A list of message dictionaries, e.g.,
                                             [{'role': 'user', 'content': 'Hello!'},
                                              {'role': 'assistant', 'content': 'Hi there!'}, ...]
            **kwargs: Additional provider-specific parameters (e.g., temperature, max_tokens).

        Returns:
            str: The text content of the model's response.

        Raises:
            NotImplementedError: If the method is not implemented by the subclass.
            Exception: Can raise exceptions related to API calls (network issues, auth errors, etc.).
        """
        pass

    def get_usage(self) -> Dict[str, Any]:
        """
        Returns the token usage information from the last API call, if available.

        Returns:
            Dict[str, Any]: A dictionary containing usage details (e.g., prompt_tokens, completion_tokens, total_tokens).
                            Returns an empty dict if usage info is not available or supported.
        """
        return self.last_usage

    def get_model_id(self) -> str:
        """
        Returns the unique identifier of the model instance.
        """
        return self.model_id

    # Optional: Add other common methods later, like:
    # @abstractmethod
    # async def achat(self, messages: List[Dict[str, str]], **kwargs) -> str:
    #     """Asynchronous version of chat."""
    #     pass

    # @abstractmethod
    # def stream_chat(self, messages: List[Dict[str, str]], **kwargs) -> Iterator[str]:
    #     """Streams the response chunks."""
    #     pass

    # @abstractmethod
    # def embed(self, text: str, **kwargs) -> List[float]:
    #     """Generates embeddings for the given text."""
    #     pass