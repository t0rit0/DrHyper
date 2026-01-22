from __future__ import annotations
from config.settings import ConfigManager
from utils.llm_loader import load_chat_model
from utils.vision_loader import load_vision_model
from typing import Optional, Tuple

# ANSI color codes for terminal output
class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    RED = "\033[91m"

def format_doctor_response(text: str) -> str:
    """Format doctor's response with color"""
    return f"{Colors.GREEN}{Colors.BOLD}DrHyper:{Colors.RESET} {text}"

def format_patient_input(text: str) -> str:
    """Format patient input with color"""
    return f"{Colors.BLUE}{Colors.BOLD}Patient:{Colors.RESET} {text}"

def format_system_message(text: str) -> str:
    """Format system messages with color"""
    return f"{Colors.YELLOW}System:{Colors.RESET} {text}"

def format_debug(text: str) -> str:
    """Format debug messages with color"""
    return f"{Colors.CYAN}Debug:{Colors.RESET} {text}"

def format_error(text: str) -> str:
    """Format error messages with color"""
    return f"{Colors.RED}Error:{Colors.RESET} {text}"

def parse_json_response(response_content: str) -> dict:
    """
    Parse JSON from response content, handling cases where the JSON might be
    enclosed in markdown code blocks.
    
    Args:
        response_content (str): The response content which may contain JSON
            directly or enclosed in ```json ... ``` code blocks
            
    Returns:
        dict: The parsed JSON data
    """
    import json
    import re
    
    # Check if the content contains markdown JSON code blocks
    json_block_pattern = r'```json\s*([\s\S]*?)\s*```'
    match = re.search(json_block_pattern, response_content)
    
    if match:
        # Extract JSON from inside the code block
        json_content = match.group(1)
    else:
        # Assume the entire content is JSON
        json_content = response_content
    
    try:
        return json.loads(json_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON: {e}. Content: {json_content[:100]}...")
    
def load_models(verbose=False):
    """Load AI models"""
    print(format_system_message("Loading AI models..."))
    config = ConfigManager()
    try:
        conv_model = load_chat_model(config.conversation_llm.provider, 
                                     config.conversation_llm.model,
                                     api_key=config.conversation_llm.api_key,
                                     base_url=config.conversation_llm.base_url,
                                     model_path=config.conversation_llm.model_path,
                                     max_tokens=config.conversation_llm.max_tokens,
                                     temperature=config.conversation_llm.temperature)
        graph_model = load_chat_model(config.graph_llm.provider,
                                      config.graph_llm.model,
                                      api_key=config.graph_llm.api_key,
                                      base_url=config.graph_llm.base_url,
                                      model_path=config.graph_llm.model_path,
                                      max_tokens=config.graph_llm.max_tokens,
                                      temperature=config.graph_llm.temperature)
        return conv_model, graph_model
    except Exception as e:
        print(format_system_message(f"Error loading models: {e}"))
        if verbose:
            import traceback
            print(format_debug(traceback.format_exc()))
        raise(e)

def load_vision_model_or_none(verbose=False) -> Optional:
    """
    Load vision model if configured, otherwise return None.

    Supports both API-based and local vision models.

    Args:
        verbose: Enable verbose logging

    Returns:
        Vision model instance or None if not configured
    """
    print(format_system_message("Checking for vision model configuration..."))
    config = ConfigManager()

    try:
        provider = config.vision_llm.provider

        # API-based model (provider = "custom")
        if provider == "custom":
            if not config.vision_llm.model or not config.vision_llm.api_key or not config.vision_llm.base_url:
                print(format_system_message("Vision API model not configured (missing model, api_key, or base_url)"))
                return None

            if config.vision_llm.api_key == "your-vision-api-key":
                print(format_system_message("Vision API model not configured (using placeholder api_key)"))
                return None

            vision_model = load_vision_model(
                provider=config.vision_llm.provider,
                model_name=config.vision_llm.model,
                api_key=config.vision_llm.api_key,
                base_url=config.vision_llm.base_url,
                max_tokens=config.vision_llm.max_tokens,
                temperature=config.vision_llm.temperature
            )

            print(format_system_message(f"Loaded API vision model: {config.vision_llm.model}"))
            return vision_model

        # Local model (provider = "local")
        elif provider == "local":
            if not config.vision_llm.model_path:
                print(format_system_message("Local vision model not configured (missing model_path)"))
                return None

            vision_model = load_vision_model(
                provider=config.vision_llm.provider,
                model_path=config.vision_llm.model_path,
                max_tokens=config.vision_llm.max_tokens,
                temperature=config.vision_llm.temperature,
                device=config.vision_llm.device
            )

            print(format_system_message(f"Loaded local vision model: {config.vision_llm.model_path}"))
            return vision_model

        else:
            print(format_system_message(f"Unsupported vision provider: {provider}"))
            return None

    except Exception as e:
        print(format_system_message(f"Error loading vision model: {e}"))
        if verbose:
            import traceback
            print(format_debug(traceback.format_exc()))
        return None

def load_models_with_vision(verbose=False) -> Tuple:
    """
    Load conversation, graph, and optionally vision models.

    Args:
        verbose: Enable verbose logging

    Returns:
        Tuple of (conv_model, graph_model, vision_model_or_none)
    """
    conv_model, graph_model = load_models(verbose)
    vision_model = load_vision_model_or_none(verbose)
    return conv_model, graph_model, vision_model
