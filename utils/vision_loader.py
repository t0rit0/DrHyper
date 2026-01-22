from typing import List, Optional, Dict, Any, Iterator, Union
from langchain_core.language_models import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from openai import OpenAI
import base64
import os
from pathlib import Path

from config.settings import ConfigManager
from utils.logging import get_logger

try:
    from transformers import AutoModelForCausalLM, AutoProcessor
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class VisionChatModel(BaseChatModel):
    """Vision Chat Model that interfaces with multi-modal LLMs via API (OpenAI-compatible)"""

    # Pydantic fields - need to be explicitly defined
    model_name: str
    api_key: str
    base_url: str
    temperature: float = 0.0
    max_tokens: int = 8192

    # Private attributes - not managed by Pydantic
    _logger: Any = None

    def __init__(self, model_name, api_key, base_url, temperature, max_tokens, **kwargs):
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )

        # Initialize logger as private attribute (not managed by Pydantic)
        object.__setattr__(self, '_logger', get_logger(self.__class__.__name__))

    @property
    def _llm_type(self) -> str:
        return "vision_api"

    def _convert_message(self, msg: BaseMessage):
        """Convert LangChain message to API format with support for multimodal content"""
        if isinstance(msg, HumanMessage):
            # msg.content can be a string or a list of content blocks
            if isinstance(msg.content, str):
                return {"role": "user", "content": msg.content}
            elif isinstance(msg.content, list):
                # Multimodal content: [{"type": "image_url", ...}, {"type": "text", ...}]
                return {"role": "user", "content": msg.content}
            else:
                # Fallback for unexpected content format
                return {"role": "user", "content": str(msg.content)}
        elif isinstance(msg, AIMessage):
            return {"role": "assistant", "content": msg.content}
        elif isinstance(msg, SystemMessage):
            return {"role": "system", "content": msg.content}
        else:
            raise ValueError(f"Unknown message type: {type(msg)}")

    def _create_client(self):
        """Create OpenAI-compatible client for vision model"""
        return OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        """Make a non-streaming API call"""
        client = self._create_client()

        # Convert messages to API format
        api_messages = [self._convert_message(msg) for msg in messages]

        self._logger.debug(f"Sending {len(api_messages)} messages to vision model")

        completion = client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=False,
            stop=stop,
        )

        response_content = completion.choices[0].message.content
        self._logger.debug(f"Received response length: {len(response_content) if response_content else 0}")

        return response_content

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResult:
        """Generate response from vision model"""
        if not stream:
            content = self._call(messages, stop=stop, **kwargs)
            generation = ChatGeneration(message=AIMessage(content=content))

            if run_manager:
                run_manager.on_llm_new_token(generation.message.content)

            return ChatResult(generations=[generation])
        else:
            # Streaming support for vision models (if needed in the future)
            content = ""
            for token in self._stream_response(messages, stop=stop, run_manager=run_manager, **kwargs):
                content += token

            generation = ChatGeneration(message=AIMessage(content=content))
            return ChatResult(generations=[generation])

    def _stream_response(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                        run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs) -> Iterator[str]:
        """Stream response from vision model (if supported)"""
        client = self._create_client()

        api_messages = [self._convert_message(msg) for msg in messages]

        stream = client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=True,
            stop=stop,
        )

        for chunk in stream:
            if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                content = chunk.choices[0].delta.content
                if content:
                    if run_manager:
                        run_manager.on_llm_new_token(content)
                    yield content


class LocalVisionChatModel(BaseChatModel):
    """
    Local Qwen-VL Chat Model for image+text understanding.

    Supports Qwen-VL models:
    - Qwen/Qwen-VL-Chat
    - Qwen/Qwen-VL-7B-Chat
    - Qwen/Qwen-VL-Chat-Int4
    - Local checkpoints of Qwen-VL models
    """

    # Pydantic fields
    model_path: str
    temperature: float = 0.0
    max_tokens: int = 2048
    device: str = "auto"
    model_kwargs: Dict[str, Any] = {}
    generation_kwargs: Dict[str, Any] = {}

    # Private attributes
    _model = None
    _tokenizer = None
    _processor = None
    _logger: Any = None

    def __init__(self, model_path: str, **kwargs):
        # Extract model_path before passing to parent
        temperature = kwargs.get('temperature', 0.0)
        max_tokens = kwargs.get('max_tokens', 2048)
        device = kwargs.get('device', 'auto')
        model_kwargs_val = kwargs.get('model_kwargs', {})
        generation_kwargs_val = kwargs.get('generation_kwargs', {})

        super().__init__(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device,
            model_kwargs=model_kwargs_val,
            generation_kwargs=generation_kwargs_val
        )

        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers and torch are required for LocalVisionChatModel. "
                "Please install them with: pip install transformers torch pillow"
            )

        # Initialize logger as private attribute
        object.__setattr__(self, '_logger', get_logger(self.__class__.__name__))

    def _load_model(self):
        """Load Qwen-VL model from the specified path"""
        self._logger.info(f"Loading Qwen-VL model from: {self.model_path}")

        try:
            # Load processor
            self._logger.debug("Loading AutoProcessor...")
            self._processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                **self.model_kwargs
            )

            # Load model
            self._logger.debug("Loading AutoModelForCausalLM...")
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                device_map=self.device,
                trust_remote_code=True,
                **self.model_kwargs
            )

            # Load tokenizer (Qwen-VL uses the processor's tokenizer)
            self._tokenizer = self._processor.tokenizer

            # Set to eval mode
            if hasattr(self._model, 'eval'):
                self._model.eval()

            self._logger.info(f"Successfully loaded Qwen-VL model on {self._model.device}")

        except Exception as e:
            self._logger.error(f"Failed to load Qwen-VL model: {e}")
            raise

    @property
    def _llm_type(self) -> str:
        return "local_qwen_vl"

    def _extract_images_from_message(self, content: Union[str, List]) -> tuple:
        """
        Extract images and text from message content.

        Returns:
            tuple: (images_list, text_string)
        """
        from PIL import Image
        import requests
        from io import BytesIO

        images = []
        text_parts = []

        if isinstance(content, str):
            # Pure text, no images
            return [], content

        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "image_url":
                    # Extract image from URL or base64
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url

                    # Convert to PIL image
                    img = self._load_image(url)
                    if img:
                        images.append(img)
                        self._logger.debug(f"Loaded image: {len(images)} total")

                elif item.get("type") == "text":
                    text_parts.append(item.get("text", ""))

        return images, " ".join(text_parts)

    def _load_image(self, url: str):
        """Load PIL image from URL or base64 data URL"""
        from PIL import Image
        import requests
        from io import BytesIO

        try:
            if url.startswith("data:image"):
                # Base64 encoded image
                if "," in url:
                    base64_data = url.split(",", 1)[1]
                else:
                    base64_data = url

                image_bytes = base64.b64decode(base64_data)
                img = Image.open(BytesIO(image_bytes))
                return img.convert("RGB")

            elif url.startswith("http://") or url.startswith("https://"):
                # URL
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content))
                return img.convert("RGB")

            else:
                # File path
                if os.path.exists(url):
                    img = Image.open(url)
                    return img.convert("RGB")
                else:
                    self._logger.warning(f"Image file not found: {url}")
                    return None

        except Exception as e:
            self._logger.error(f"Failed to load image from {url[:50]}...: {e}")
            return None

    def _format_messages_for_model(self, messages: List[BaseMessage]) -> tuple:
        """
        Format messages for Qwen-VL model.

        Returns:
            tuple: (images, text_prompt)
        """
        all_images = []
        all_text = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                images, text = self._extract_images_from_message(msg.content)
                all_images.extend(images)
                all_text.append(text)
            elif isinstance(msg, SystemMessage):
                all_text.append(f"[System]: {msg.content}")
            elif isinstance(msg, AIMessage):
                all_text.append(f"[Assistant]: {msg.content}")

        return all_images, "\n".join(all_text)

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs) -> str:
        """Generate response from Qwen-VL model"""
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call _load_model() first.")

        # Extract images and format prompt
        images, text_prompt = self._format_messages_for_model(messages)

        self._logger.debug(f"Processing with {len(images)} image(s), prompt length: {len(text_prompt)}")

        # Format query for Qwen-VL
        if images:
            # Qwen-VL format: use from_list_format
            query = self._tokenizer.from_list_format([
                {'image': images[0]},  # Qwen-VL typically handles one image at a time
                {'text': text_prompt},
            ])
        else:
            query = text_prompt

        # Generate response using Qwen-VL's chat interface
        response, _ = self._model.chat(
            self._processor,
            query=query,
            history=None,
            max_length=self.max_tokens,
            temperature=self.temperature if self.temperature > 0 else 0.7,
            generation_config=self.generation_kwargs
        )

        return response

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: bool = False,
        **kwargs
    ) -> ChatResult:
        """Generate response"""
        content = self._call(messages, stop=stop, **kwargs)
        generation = ChatGeneration(message=AIMessage(content=content))

        if run_manager:
            run_manager.on_llm_new_token(generation.message.content)

        return ChatResult(generations=[generation])


def load_vision_model(
    provider: str = "custom",
    model_name: str = "",
    model_path: str = "",
    api_key: str = "",
    base_url: str = "",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    device: str = "auto"
) -> Optional[Union[VisionChatModel, LocalVisionChatModel]]:
    """
    Load a vision model by its configuration.

    Args:
        provider: Model provider ("custom" for API, "local" for Qwen-VL)
        model_name: Model identifier (for API models)
        model_path: Path to local Qwen-VL model or HuggingFace model ID
        api_key: API key for the service (for API models)
        base_url: API endpoint URL (for API models)
        temperature: Sampling temperature
        max_tokens: Maximum tokens in response
        device: Device for local models ("cpu", "cuda", "auto")

    Returns:
        VisionChatModel (API) or LocalVisionChatModel (Qwen-VL) instance

    Example:
        # API model
        model = load_vision_model(
            provider="custom",
            model_name="qwen-vl-max",
            api_key="sk-xxx",
            base_url="https://api.openai.com/v1"
        )

        # Local Qwen-VL model
        model = load_vision_model(
            provider="local",
            model_path="Qwen/Qwen-VL-7B-Chat",
            device="cuda"
        )
    """
    logger = get_logger("VisionLoader")

    if provider == "custom":
        # API-based vision model
        if not model_name or not api_key or not base_url:
            logger.warning("Vision API model not configured (missing model_name, api_key, or base_url)")
            return None

        logger.info(f"Loading API vision model: {model_name}")
        return VisionChatModel(
            model_name=model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens
        )

    elif provider == "local":
        # Local Qwen-VL model
        if not model_path:
            logger.warning("Local vision model not configured (missing model_path)")
            return None

        logger.info(f"Loading local Qwen-VL model from: {model_path}")
        return LocalVisionChatModel(
            model_path=model_path,
            temperature=temperature,
            max_tokens=max_tokens,
            device=device
        )

    else:
        logger.error(f"Unsupported vision provider: {provider}")
        raise ValueError(
            f"Vision provider '{provider}' not supported. "
            "Use 'custom' for API-based models or 'local' for local Qwen-VL models."
        )
