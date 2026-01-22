import os
import uuid
import json
import base64
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from datetime import datetime
from langchain.schema import HumanMessage

from config.settings import ConfigManager
from utils.vision_loader import load_vision_model, VisionChatModel
from utils.logging import get_logger


class ImageStorage:
    """Handle temporary storage of images"""

    def __init__(self, storage_dir: Optional[str] = None):
        self.config = ConfigManager()
        self.storage_dir = storage_dir or os.path.join(
            self.config.system.working_directory,
            "images"
        )
        self._ensure_storage_dir()

    def _ensure_storage_dir(self):
        """Ensure storage directory exists"""
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

    def save_base64_image(self, image_data: str, filename: Optional[str] = None) -> str:
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        if not filename:
            filename = f"img_{uuid.uuid4().hex[:12]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        filepath = os.path.join(self.storage_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        return filepath

    def get_image_as_base64(self, filepath: str) -> str:
        with open(filepath, "rb") as f:
            image_bytes = f.read()

        # Get MIME type
        ext = Path(filepath).suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(ext, "image/png")

        # Encode to base64
        base64_data = base64.b64encode(image_bytes).decode("utf-8")

        return f"data:{mime_type};base64,{base64_data}"

    def cleanup_old_images(self, days: int = 7):
        """
        Remove images older than given days.
        """
        import time

        now = time.time()
        cutoff = now - (days * 86400)  # 86400 seconds in a day

        removed_count = 0
        for filename in os.listdir(self.storage_dir):
            filepath = os.path.join(self.storage_dir, filename)
            if os.path.isfile(filepath):
                file_mtime = os.path.getmtime(filepath)
                if file_mtime < cutoff:
                    os.remove(filepath)
                    removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} old image(s)")


class ImageAnalyzer:
    """
    Analyze medical images using multi-modal LLMs.

    This class handles:
    - Loading and managing vision models
    - Processing images (base64, URLs, file paths)
    - Generating and executing analysis queries
    - Managing analysis logs and results
    """

    def __init__(
        self,
        vision_model: Optional[VisionChatModel] = None,
        storage_dir: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the ImageAnalyzer.

        Args:
            vision_model: Pre-configured VisionChatModel instance. If None, loads from config
            storage_dir: Directory for temporary image storage
            verbose: Enable verbose logging
        """
        self.config = ConfigManager()
        self.logger = get_logger(self.__class__.__name__)

        if verbose:
            self.logger.setLevel("DEBUG")

        # Initialize vision model
        if vision_model:
            self.vision_model = vision_model
            self.logger.info("Using provided vision model")
        else:
            # Load from configuration
            try:
                vision_config = self.config.vision_llm if hasattr(self.config, 'vision_llm') else None

                if vision_config and vision_config.model:  # Check both config exists and model is set
                    self.vision_model = load_vision_model(
                        provider=vision_config.provider,
                        model_name=vision_config.model,
                        api_key=vision_config.api_key,
                        base_url=vision_config.base_url,
                        temperature=vision_config.temperature,
                        max_tokens=vision_config.max_tokens
                    )
                    if self.vision_model:
                        self.logger.info(f"Loaded vision model: {vision_config.model}")
                    else:
                        self.logger.warning("Failed to load vision model")
                        self.vision_model = None
                else:
                    self.logger.warning("No vision LLM configuration found")
                    self.vision_model = None
            except Exception as e:
                self.logger.error(f"Failed to load vision model: {e}")
                self.vision_model = None

        # Initialize image storage
        self.storage = ImageStorage(storage_dir)

        # Analysis history
        self.analysis_history: List[Dict[str, Any]] = []

    def is_available(self) -> bool:
        """Check if vision model is available"""
        return self.vision_model is not None

    def analyze(
        self,
        query: str,
        images: List[str],
        image_type: str = "base64",
        conversation_context: Optional[str] = None
    ) -> Tuple[str, List[str]]:
        """
        Analyze images with a specific query.

        Args:
            query: The analysis query/question for the vision model
            images: List of images (base64 strings, URLs, or file paths)
            image_type: Type of image input: "base64", "url", or "path"
            conversation_context: Optional conversation context to include

        Returns:
            Tuple of (analysis_report, log_messages)

        Raises:
            ValueError: If vision model is not available or invalid image_type
        """
        log_messages = []

        if not self.is_available():
            error_msg = "Vision model is not available. Please configure vision LLM settings."
            self.logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)

        self.logger.info(f"Starting image analysis: {len(images)} image(s), image_type={image_type}")
        log_messages.append(f"Starting image analysis with {len(images)} image(s)")

        # Prepare image content
        try:
            message_content = self._prepare_message_content(images, image_type, query, conversation_context)
            log_messages.append(f"Prepared message content with {len([c for c in message_content if c.get('type') == 'image_url'])} image(s)")
        except Exception as e:
            error_msg = f"Failed to prepare message content: {e}"
            self.logger.error(error_msg)
            log_messages.append(error_msg)
            raise ValueError(error_msg)

        # Log the query
        self.logger.debug(f"Analysis query: {query[:200]}...")
        log_messages.append(f"Query: {query[:100]}...")

        # Invoke vision model
        try:
            self.logger.info("Invoking vision model...")
            log_messages.append("Invoking vision model...")

            response = self.vision_model.invoke([HumanMessage(content=message_content)])

            analysis_report = response.content
            self.logger.info(f"Received analysis report, length: {len(analysis_report)}")
            log_messages.append(f"Received analysis report, length: {len(analysis_report)}")

            # Store in history
            self._store_analysis(query, images, image_type, analysis_report)

            return analysis_report, log_messages

        except Exception as e:
            error_msg = f"Vision model invocation failed: {e}"
            self.logger.error(error_msg)
            log_messages.append(error_msg)
            raise

    def _prepare_message_content(
        self,
        images: List[str],
        image_type: str,
        query: str,
        conversation_context: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare the message content for vision model.

        Args:
            images: List of images
            image_type: Type of image input
            query: Analysis query
            conversation_context: Optional context

        Returns:
            List of content blocks for the message
        """
        message_content = []

        # Add images
        for i, img in enumerate(images):
            if image_type == "base64":
                # Already in base64 format
                if img.startswith("data:"):
                    # Full data URL
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": img}
                    })
                else:
                    # Raw base64, need to add prefix
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img}"}
                    })
            elif image_type == "url":
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": img}
                })
            elif image_type == "path":
                # Read file and convert to base64
                base64_data = self.storage.get_image_as_base64(img)
                message_content.append({
                    "type": "image_url",
                    "image_url": {"url": base64_data}
                })
            else:
                raise ValueError(f"Invalid image_type: {image_type}. Must be 'base64', 'url', or 'path'")

            self.logger.debug(f"Added image {i+1}/{len(images)} to message content")

        # Add conversation context if provided
        if conversation_context:
            message_content.append({
                "type": "text",
                "text": f"**Conversation Context**:\n{conversation_context}\n\n"
            })

        # Add query
        message_content.append({
            "type": "text",
            "text": query
        })

        return message_content

    def _store_analysis(self, query: str, images: List[str], image_type: str, report: str):
        """Store analysis in history"""
        analysis_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "image_count": len(images),
            "image_type": image_type,
            "report_length": len(report),
            "report": report
        }
        self.analysis_history.append(analysis_entry)

        # Keep only last 100 analyses
        if len(self.analysis_history) > 100:
            self.analysis_history = self.analysis_history[-100:]

    def get_analysis_history(self, last_n: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent analysis history.

        Args:
            last_n: Number of recent analyses to return

        Returns:
            List of analysis entries (without full report to save space)
        """
        history = self.analysis_history[-last_n:]
        # Return summary without full report
        return [
            {
                "timestamp": entry["timestamp"],
                "query": entry["query"],
                "image_count": entry["image_count"],
                "report_length": entry["report_length"]
            }
            for entry in history
        ]
