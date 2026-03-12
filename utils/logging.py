import logging
import os
import json
import threading
from contextlib import contextmanager
from typing import Optional, Dict, Any, List
from datetime import datetime


class ConversationContext:
    """Thread-local context for conversation tracing"""
    _local = threading.local()
    
    @classmethod
    def set(cls, conversation_id: str, patient_id: Optional[str] = None):
        cls._local.conversation_id = conversation_id
        cls._local.patient_id = patient_id
    
    @classmethod
    def get(cls) -> Dict[str, Optional[str]]:
        return {
            "conversation_id": getattr(cls._local, "conversation_id", None),
            "patient_id": getattr(cls._local, "patient_id", None)
        }
    
    @classmethod
    def clear(cls):
        if hasattr(cls._local, "conversation_id"):
            del cls._local.conversation_id
        if hasattr(cls._local, "patient_id"):
            del cls._local.patient_id


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging (machine-readable)"""
    
    def __init__(self, include_context: bool = True):
        super().__init__()
        self.include_context = include_context
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add conversation context if available
        if self.include_context:
            context = ConversationContext.get()
            if context["conversation_id"]:
                log_entry["conversation_id"] = context["conversation_id"]
            if context["patient_id"]:
                log_entry["patient_id"] = context["patient_id"]
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ('name', 'msg', 'args', 'created', 'filename', 'funcName', 
                          'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                          'pathname', 'process', 'processName', 'relativeCreated',
                          'stack_info', 'exc_info', 'exc_text', 'thread', 'threadName',
                          'message', 'asctime'):
                log_entry[key] = value
        
        return json.dumps(log_entry, ensure_ascii=False, default=str)


class DetailedFormatter(logging.Formatter):
    """Human-readable detailed formatter with context support"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    def format(self, record: logging.LogRecord) -> str:
        # Add conversation context as prefix if available
        context = ConversationContext.get()
        if context["conversation_id"]:
            ctx_str = f"[conv:{context['conversation_id'][:8]}]"
            if context["patient_id"]:
                ctx_str += f"[patient:{context['patient_id']}]"
            record.msg = f"{ctx_str} {record.msg}"
        return super().format(record)


# Global configuration
_log_config: Dict[str, Any] = {
    "level": logging.INFO,
    "file_level": logging.DEBUG,
    "log_dir": "logs",
    "max_bytes": 10 * 1024 * 1024,  # 10MB
    "backup_count": 5,
    "enable_structured": False,  # Enable JSON structured logging
}


def configure_logging(
    level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    log_dir: str = "logs",
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
    enable_structured: bool = False
):
    """
    Configure global logging settings
    
    Args:
        level: Console log level
        file_level: File log level (more detailed)
        log_dir: Directory for log files
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup log files to keep
        enable_structured: Enable JSON structured logging for machine parsing
    """
    global _log_config
    _log_config.update({
        "level": level,
        "file_level": file_level,
        "log_dir": log_dir,
        "max_bytes": max_bytes,
        "backup_count": backup_count,
        "enable_structured": enable_structured,
    })


@contextmanager
def conversation_context(conversation_id: str, patient_id: Optional[str] = None):
    """
    Context manager for conversation-scoped logging
    
    All logs within this context will automatically include conversation_id
    and patient_id for easy tracing.
    
    Usage:
        with conversation_context(conv_id, patient_id):
            logger.info("Processing message")
    """
    ConversationContext.set(conversation_id, patient_id)
    try:
        yield
    finally:
        ConversationContext.clear()


def get_logger(name: str) -> logging.Logger:
    """Get or create a logger with standard configuration"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)  # Accept all levels, handlers filter
        
        # Create log directory
        log_dir = _log_config["log_dir"]
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Console handler with detailed formatter
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_log_config["level"])
        console_handler.setFormatter(DetailedFormatter())
        
        # File handler with rotation
        from logging.handlers import RotatingFileHandler
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, "server.log"),
            mode='a',
            encoding='utf-8',
            maxBytes=_log_config["max_bytes"],
            backupCount=_log_config["backup_count"]
        )
        file_handler.setLevel(_log_config["file_level"])
        file_handler.setFormatter(DetailedFormatter())
        
        # Structured JSON log file (for machine parsing/backend debugging)
        if _log_config["enable_structured"]:
            structured_handler = RotatingFileHandler(
                os.path.join(log_dir, "server_structured.log"),
                mode='a',
                encoding='utf-8',
                maxBytes=_log_config["max_bytes"],
                backupCount=_log_config["backup_count"]
            )
            structured_handler.setLevel(_log_config["file_level"])
            structured_handler.setFormatter(StructuredFormatter())
            logger.addHandler(structured_handler)
        
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger


def log_event(
    logger: logging.Logger,
    event_type: str,
    message: str,
    level: int = logging.INFO,
    **extra_data
):
    """
    Log a structured event with extra context
    
    Args:
        logger: Logger instance
        event_type: Type of event (e.g., "CONVERSATION_START", "TOOL_CALL")
        message: Human-readable message
        level: Log level
        **extra_data: Additional context data
    """
    extra = {"event_type": event_type, **extra_data}
    logger.log(level, message, extra=extra)