# models/model_router.py
import importlib
import logging
import time
from typing import List, Dict, Optional, Type, Set, Tuple

from .base_model import BaseModel
from .model_registry import MODEL_REGISTRY, get_model_metadata

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Helper Function to Dynamically Import Connectors ---
def _import_connector_class(class_name: str) -> Optional[Type[BaseModel]]:
    """Dynamically imports a connector class from the models directory."""
    try:
        module = importlib.import_module(f".{class_name.lower().replace('connector', '_connector')}", package="models")
        connector_class = getattr(module, class_name)
        if not issubclass(connector_class, BaseModel):
             logger.error(f"Class {class_name} is not a subclass of BaseModel.")
             return None
        return connector_class
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import connector class '{class_name}': {e}", exc_info=True)
        return None
# ---------------------------------------------------------

class ModelRouter:
    """
    Manages language model instances, handles model selection based on preferences
    and capabilities, and implements fallback logic.
    """

    def __init__(self):
        """
        Initializes the ModelRouter, loading and preparing connectors
        based on the MODEL_REGISTRY and available API keys.
        """
        self.models: Dict[str, BaseModel] = {} # Stores active model instances {model_id: instance}
        self._initialize_models()

    def _initialize_models(self):
        """
        Instantiates model connectors listed in MODEL_REGISTRY if their API keys are found.
        """
        logger.info("Initializing available models based on registry and API keys...")
        initialized_count = 0
        for model_id, meta in MODEL_REGISTRY.items():
            connector_class_name = meta.get("connector_class")
            api_model_name = meta.get("api_model_name", model_id) # Use specific API name if provided

            if not connector_class_name:
                logger.warning(f"Skipping model '{model_id}': 'connector_class' not defined in registry.")
                continue

            # Dynamically import the connector class
            ConnectorClass = _import_connector_class(connector_class_name)
            if not ConnectorClass:
                 logger.warning(f"Skipping model '{model_id}': Could not import connector class '{connector_class_name}'.")
                 continue

            # Try to instantiate (this will check for API keys within the connector's __init__)
            try:
                # Pass the specific API model name expected by the provider
                instance = ConnectorClass(model_id=api_model_name)
                self.models[model_id] = instance # Store instance using the internal model_id
                logger.info(f"Successfully initialized model: '{model_id}' (using {connector_class_name} for {api_model_name})")
                initialized_count += 1
            except ValueError as e: # Typically raised if API key is missing
                 logger.warning(f"Skipping model '{model_id}': Configuration error during initialization - {e}")
            except Exception as e:
                logger.error(f"Failed to initialize model '{model_id}' with connector '{connector_class_name}': {e}", exc_info=True)

        logger.info(f"ModelRouter initialization complete. {initialized_count} models loaded.")
        if initialized_count == 0:
             logger.warning("No models were loaded. Check API keys and registry configuration.")

    def get_available_models(self) -> List[str]:
        """Returns a list of successfully initialized model IDs."""
        return list(self.models.keys())

    def _select_model(self,
                      preferences: Optional[List[str]] = None,
                      required_capabilities: Optional[List[str]] = None,
                      excluded_ids: Optional[Set[str]] = None) -> Optional[Tuple[str, BaseModel]]:
        """
        Internal logic to select the best available model based on criteria.

        Args:
            preferences: Ordered list of preferred model IDs.
            required_capabilities: List of capabilities the model must have.
            excluded_ids: Set of model IDs to exclude from selection (e.g., previously failed models).

        Returns:
            A tuple (model_id, model_instance) or None if no suitable model is found.
        """
        candidates = []
        available_models = self.get_available_models()
        excluded_ids = excluded_ids or set()

        # Filter models based on availability, exclusions, and required capabilities
        for model_id in available_models:
            if model_id in excluded_ids:
                continue

            meta = get_model_metadata(model_id)
            if not meta: # Should not happen if initialized correctly, but safety check
                 continue


            # Check capabilities
            if required_capabilities:
                model_caps = meta.get("capabilities", [])
                if not all(cap in model_caps for cap in required_capabilities):
                    continue # Doesn't have all required capabilities

            # Add valid candidate with its priority
            priority = meta.get("preferred_priority", 99) # Default to low priority
            candidates.append((priority, model_id))

        if not candidates:
            logger.warning("No available models match the required criteria.")
            return None

        # Sort candidates by priority (lower number is higher priority)
        candidates.sort()

        # Now, try to match against preferences if provided
        if preferences:
             for pref_id in preferences:
                 for priority, model_id in candidates:
                      if model_id == pref_id:
                           logger.debug(f"Selected model '{model_id}' based on preference.")
                           return model_id, self.models[model_id]


        # If no preference matched or no preferences given, return the highest priority candidate
        best_priority, best_model_id = candidates[0]
        logger.debug(f"Selected model '{best_model_id}' based on highest priority ({best_priority}).")
        return best_model_id, self.models[best_model_id]


    def chat(self,
             messages: List[Dict[str, str]],
             preferences: Optional[List[str]] = None,
             required_capabilities: Optional[List[str]] = None,
             **kwargs) -> str:
        """
        Routes the chat request to the best available model based on preferences
        and capabilities, handling fallbacks on failure.

        Args:
            messages: The list of messages for the chat conversation.
            preferences: An ordered list of preferred model IDs (e.g., ["gpt-4-turbo", "claude-3-opus"]).
            required_capabilities: A list of capabilities the chosen model must possess (e.g., ["code", "vision"]).
            **kwargs: Additional arguments to pass to the underlying model's chat method.

        Returns:
            The response string from the selected model.

        Raises:
            RuntimeError: If no suitable model is found or all attempted models fail.
        """
        attempted_models: Set[str] = set()
        current_preferences = preferences or [] # Use provided preferences or empty list

        while True: # Loop to handle fallbacks
            selection = self._select_model(
                preferences=current_preferences,
                required_capabilities=required_capabilities,
                excluded_ids=attempted_models
            )

            if not selection:
                logger.error(f"No suitable models found or all options exhausted. Attempted: {attempted_models}")
                raise RuntimeError("Failed to get response: No suitable models available or all attempts failed.")

            selected_model_id, selected_model = selection
            attempted_models.add(selected_model_id)
            logger.info(f"Attempting chat with model: '{selected_model_id}'")

            try:
                start_time = time.time()
                response = selected_model.chat(messages, **kwargs)
                end_time = time.time()
                logger.info(f"Successfully received response from '{selected_model_id}' in {end_time - start_time:.2f}s.")
                # Optional: Log usage if needed here
                # logger.info(f"Usage for '{selected_model_id}': {selected_model.get_usage()}")
                return response # Success!

            except (ConnectionError, PermissionError, TimeoutError, RuntimeError) as e:
                 # These are errors typically indicating the model/API failed
                 logger.warning(f"Model '{selected_model_id}' failed: {type(e).__name__} - {e}")
                 # Find fallbacks for the failed model
                 meta = get_model_metadata(selected_model_id)
                 fallbacks = meta.get("fallbacks", []) if meta else []

                 # Use fallbacks as the next set of preferences
                 current_preferences = fallbacks
                 if not current_preferences:
                      logger.warning(f"No more fallbacks defined for '{selected_model_id}'. Trying next highest priority model.")
                      # If no fallbacks, the loop will select the next best based on priority


            except Exception as e:
                 # Catch any other unexpected error from the connector
                 logger.error(f"Unexpected error during chat with model '{selected_model_id}': {e}", exc_info=True)
                 # Treat as failure, attempt fallback/next model
                 meta = get_model_metadata(selected_model_id)
                 fallbacks = meta.get("fallbacks", []) if meta else []
                 current_preferences = fallbacks
                 if not current_preferences:
                     logger.warning(f"No more fallbacks defined for '{selected_model_id}' after unexpected error. Trying next highest priority model.")



# Example Usage
if __name__ == "__main__":
    print("\n--- Model Router Test ---")
    router = ModelRouter()
    available = router.get_available_models()
    print(f"Available models: {available}")

    if not available:
        print("Cannot run tests - no models loaded. Check API keys in settings/api_keys.env")
    else:
        test_messages = [{"role": "user", "content": "Write a short poem about the future of AI agents."}]

        print("\nTesting with default preferences:")
        try:
            response_default = router.chat(test_messages)
            print(f"Default Response: {response_default[:150]}...")
        except Exception as e:
            print(f"Default Test Failed: {e}")

        print("\nTesting with preference for Claude Haiku (likely low priority):")
        try:
            response_haiku = router.chat(test_messages, preferences=["claude-3-haiku", "gpt-3.5-turbo"])
            print(f"Haiku Preference Response: {response_haiku[:150]}...")
        except Exception as e:
             print(f"Haiku Preference Test Failed: {e}")

        print("\nTesting with preference for highest tier (GPT-4/Opus):")
        try:
            response_premium = router.chat(test_messages, preferences=["gpt-4-turbo", "claude-3-opus"])
            print(f"Premium Preference Response: {response_premium[:150]}...")
        except Exception as e:
             print(f"Premium Preference Test Failed: {e}")

        print("\nTesting with required capability 'code' (should pick a capable model):")
        try:
            response_code = router.chat(
                [{"role": "user", "content": "Write a python function to calculate factorial."}],
                required_capabilities=["code"]
            )
            print(f"'Code' Capability Response:\n{response_code}")
        except Exception as e:
            print(f"'Code' Capability Test Failed: {e}")

        # To test fallback, you might temporarily disable a preferred model's API key
        # or force an error in one of the connectors.

    print("-------------------------")