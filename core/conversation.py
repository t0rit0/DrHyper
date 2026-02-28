import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from langchain.schema import AIMessage, SystemMessage, HumanMessage, BaseMessage
from html.parser import HTMLParser

from drhyper.config.settings import ConfigManager
from .graph import EntityGraph
from .image_analyzer import ImageAnalyzer
from drhyper.prompts.templates import ConversationPrompts
from drhyper.utils.logging import get_logger

class ThinkParser(HTMLParser):
    """Parser for extracting think tags from AI responses"""
    def __init__(self):
        super().__init__()
        self.think_content = []
        self.clean_content = []
        self.in_think_tag = False
        self.all_content = []
        self.found_closing_think = False

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "think":
            self.in_think_tag = True

    def handle_endtag(self, tag):
        if tag.lower() == "think":
            self.found_closing_think = True
            self.in_think_tag = False

    def handle_data(self, data):
        self.all_content.append(data)
        if self.in_think_tag:
            self.think_content.append(data)
        else:
            self.clean_content.append(data)

class BaseConversation:
    """Base conversation class with common functionality"""
    
    def __init__(self, chat_model, max_tokens: int = 8192):
        self.config = ConfigManager()
        self.chat_model = chat_model
        self.messages: List[BaseMessage] = []
        self.think_history: List[Dict[str, Any]] = []
        self.max_tokens = max_tokens
        self.logger = get_logger(self.__class__.__name__)
        
    def _process_response(self, response_text: str) -> Dict[str, str]:
        """Extract think content and regular response from AI response"""
        parser = ThinkParser()
        parser.feed(response_text)
        
        # Handle case where there's a closing </think> tag but no opening tag
        if parser.found_closing_think and not parser.think_content:
            end_think_index = response_text.lower().find("</think>")
            if end_think_index > 0:
                parser.think_content = [response_text[:end_think_index]]
                parser.clean_content = [response_text[end_think_index + 8:]]
        
        return {
            "response": "".join(parser.clean_content).strip(),
            "think": "".join(parser.think_content).strip()
        }

class LongConversation(BaseConversation):
    """Long conversation with graph-based entity tracking"""
    
    def __init__(
        self,
        target: str,
        conv_model,
        graph_model,
        routine: Optional[str] = None,
        visualize: bool = False,
        working_directory: Optional[str] = None,
        stream: bool = False,
        **graph_params
    ):
        super().__init__(conv_model)
        self.target = target
        self.graph_model = graph_model
        self.routine = routine
        self.visualize = visualize
        self.working_directory = working_directory or self.config.system.working_directory
        self.stream = stream
        
        # Initialize entity graph with configuration parameters
        self.plan_graph = EntityGraph(
            target=target,
            graph_model=graph_model,
            conv_model=conv_model,
            routine=routine,
            visualize=visualize,
            working_directory=self.working_directory,
            node_hit_threshold=graph_params.get('node_hit_threshold', self.config.system.node_hit_threshold),
            confidential_threshold=graph_params.get('confidential_threshold', self.config.system.confidential_threshold),
            relevance_threshold=graph_params.get('relevance_threshold', self.config.system.relevance_threshold),
            weight_threshold=graph_params.get('weight_threshold', self.config.system.weight_threshold),
            alpha=graph_params.get('alpha', self.config.system.alpha),
            beta=graph_params.get('beta', self.config.system.beta),
            gamma=graph_params.get('gamma', self.config.system.gamma)
        )
        
        self.current_hint = ""
        self.messages: List[BaseMessage] = []
        self.entire_messages: List[BaseMessage] = []
        self.think_history: List[Dict[str, Any]] = []
        self.message_reserve_turns: int = 2  # Number of turns to reserve in history

        # Initialize image analyzer for medical image analysis
        self.image_analyzer = ImageAnalyzer(verbose=False)

        # Initialize conversation prompts for image analysis
        self.conv_prompts = ConversationPrompts()

        self._ensure_working_directory()
    
    def _ensure_working_directory(self):
        """Ensure working directory exists"""
        log_messages = []
        if self.working_directory and not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
            self.logger.info(f"Created working directory: {self.working_directory}")
            # log_messages.append(f"Created working directory: {self.working_directory}")
        else:
            self.logger.info("Working directory already exists or not specified")
            # log_messages.append("Working directory already exists or not specified")
        return log_messages

    def to_cache_dict(self):
        """
        Convert conversation state to a lightweight dictionary for caching.
        This excludes heavy objects like EntityGraph and ImageAnalyzer.

        Returns:
            Dictionary containing only serializable conversation state
        """
        from datetime import datetime
        import networkx as nx

        # Convert LangChain BaseMessage objects to dict
        def message_to_dict(msg):
            if hasattr(msg, 'type'):
                return {
                    "type": msg.type,
                    "content": msg.content
                }
            return {
                "type": msg.__class__.__name__,
                "content": msg.content
            }

        # Helper function to convert datetime objects to ISO strings
        def make_json_serializable(obj):
            """Recursively convert datetime objects to ISO format strings."""
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            return obj

        # Serialize NetworkX graphs to JSON-compatible format
        # First get the raw data, then convert datetime objects
        entity_graph_raw = nx.node_link_data(self.plan_graph.entity_graph, edges="links")
        relation_graph_raw = nx.node_link_data(self.plan_graph.relation_graph, edges="links")

        # Convert datetime objects in nodes and links
        entity_graph_data = {
            "nodes": [make_json_serializable(dict(node)) for node in entity_graph_raw.get("nodes", [])],
            "links": [make_json_serializable(dict(link)) for link in entity_graph_raw.get("links", [])],
            "directed": entity_graph_raw.get("directed", True),
            "multigraph": entity_graph_raw.get("multigraph", False),
            "graph": entity_graph_raw.get("graph", {})
        }
        relation_graph_data = {
            "nodes": [make_json_serializable(dict(node)) for node in relation_graph_raw.get("nodes", [])],
            "links": [make_json_serializable(dict(link)) for link in relation_graph_raw.get("links", [])],
            "directed": relation_graph_raw.get("directed", True),
            "multigraph": relation_graph_raw.get("multigraph", False),
            "graph": relation_graph_raw.get("graph", {})
        }

        return {
            "messages": [message_to_dict(m) for m in self.messages],
            "entire_messages": [message_to_dict(m) for m in self.entire_messages],
            "current_hint": self.current_hint,
            "step": getattr(self, 'step', 0),
            "think_history": self.think_history,
            "message_reserve_turns": self.message_reserve_turns,
            "target": self.target,
            "routine": self.routine,
            "visualize": self.visualize,
            "working_directory": self.working_directory,
            "stream": self.stream,
            "entity_graph": entity_graph_data,  # Serialize graph to JSON
            "relation_graph": relation_graph_data,  # Serialize graph to JSON
            "graph_state": {
                "step": self.plan_graph.step,
                "accomplish": self.plan_graph.accomplish,
                "prev_node": self.plan_graph.prev_node
            },
            "metadata": {
                "cached_at": datetime.now().isoformat(),
                "version": "2.2",  # Updated version with datetime serialization
                "message_count": len(self.messages),
                "entity_graph_nodes": len(entity_graph_data['nodes']),
                "entity_graph_edges": len(entity_graph_data['links']),
                "relation_graph_nodes": len(relation_graph_data['nodes']),
                "relation_graph_edges": len(relation_graph_data['links'])
            }
        }

    @classmethod
    def from_cache_dict(
        cls,
        cache_dict: dict,
        conv_model,
        graph_model
    ):
        """
        Restore conversation from cached dictionary.

        Args:
            cache_dict: Cached conversation state dictionary
            conv_model: Conversation LLM model
            graph_model: Graph LLM model

        Returns:
            Restored LongConversation instance
        """
        from langchain.schema import HumanMessage, AIMessage, SystemMessage
        import networkx as nx

        # Create new instance with minimal initialization
        instance = cls.__new__(cls)

        # Set basic attributes
        instance.config = ConfigManager()
        instance.logger = get_logger(cls.__name__)
        instance.chat_model = conv_model
        instance.target = cache_dict.get("target", "")
        instance.routine = cache_dict.get("routine")
        instance.visualize = cache_dict.get("visualize", False)
        instance.working_directory = cache_dict.get("working_directory")
        instance.stream = cache_dict.get("stream", False)
        instance.message_reserve_turns = cache_dict.get("message_reserve_turns", 2)

        # Restore conversation state
        instance.current_hint = cache_dict.get("current_hint", "")
        instance.step = cache_dict.get("step", 0)
        instance.think_history = cache_dict.get("think_history", [])

        # Convert dict messages back to LangChain BaseMessage objects
        def dict_to_message(msg_dict):
            msg_type = msg_dict.get("type", "")
            content = msg_dict.get("content", "")

            if msg_type == "human":
                return HumanMessage(content=content)
            elif msg_type == "ai":
                return AIMessage(content=content)
            else:  # system or other
                return SystemMessage(content=content)

        instance.messages = [dict_to_message(m) for m in cache_dict.get("messages", [])]
        instance.entire_messages = [dict_to_message(m) for m in cache_dict.get("entire_messages", [])]

        # Initialize EntityGraph (creates empty graphs)
        instance.plan_graph = EntityGraph(
            target=instance.target,
            graph_model=graph_model,
            conv_model=conv_model,
            routine=instance.routine,
            visualize=instance.visualize,
            working_directory=instance.working_directory,
            # Use default parameters from config
            node_hit_threshold=instance.config.system.node_hit_threshold,
            confidential_threshold=instance.config.system.confidential_threshold,
            relevance_threshold=instance.config.system.relevance_threshold,
            weight_threshold=instance.config.system.weight_threshold,
            alpha=instance.config.system.alpha,
            beta=instance.config.system.beta,
            gamma=instance.config.system.gamma
        )

        # Restore graph structures from cache (v2.1+)
        if "entity_graph" in cache_dict and "relation_graph" in cache_dict:
            instance.logger.info("Restoring graph structures from cache...")
            entity_graph_data = cache_dict["entity_graph"]
            relation_graph_data = cache_dict["relation_graph"]

            # Determine edge key name (compatibility with different versions)
            # Old format: uses 'edges'
            # New format: uses 'links'
            edge_key = "links" if "links" in entity_graph_data else "edges"
            instance.logger.info(f"Using edge key: '{edge_key}'")

            # Helper function to convert ISO strings back to datetime objects
            def parse_datetime_strings(obj):
                """Recursively convert ISO datetime strings to datetime objects."""
                from datetime import datetime
                if isinstance(obj, dict):
                    result = {}
                    for k, v in obj.items():
                        # Check for common datetime field names
                        if k in ('extracted_at', 'last_updated_at') and isinstance(v, str):
                            try:
                                result[k] = datetime.fromisoformat(v)
                            except (ValueError, TypeError):
                                result[k] = v
                        else:
                            result[k] = parse_datetime_strings(v)
                    return result
                elif isinstance(obj, (list, tuple)):
                    return [parse_datetime_strings(item) for item in obj]
                return obj

            # Convert datetime strings in node data
            entity_graph_data = parse_datetime_strings(entity_graph_data)
            relation_graph_data = parse_datetime_strings(relation_graph_data)

            # Deserialize NetworkX graphs
            instance.plan_graph.entity_graph = nx.node_link_graph(entity_graph_data, edges=edge_key)
            instance.plan_graph.relation_graph = nx.node_link_graph(relation_graph_data, edges=edge_key)

            # Restore graph state
            graph_state = cache_dict.get("graph_state", {})
            instance.plan_graph.step = graph_state.get("step", 0)
            instance.plan_graph.accomplish = graph_state.get("accomplish", False)
            instance.plan_graph.prev_node = graph_state.get("prev_node")

            instance.logger.info(f"Graphs restored: {instance.plan_graph.entity_graph.number_of_nodes()} nodes, "
                               f"{instance.plan_graph.entity_graph.number_of_edges()} edges")
        else:
            instance.logger.warning("No graph data in cache (v2.0 format), using empty graphs")
            instance.logger.info("Graph state will be empty until first message")

        # Recreate ImageAnalyzer with current config
        instance.image_analyzer = ImageAnalyzer(verbose=False)
        instance.conv_prompts = ConversationPrompts()

        metadata = cache_dict.get('metadata', {})
        version = metadata.get('version', 'unknown')
        instance.logger.info(f"Conversation restored from cache v{version}")

        return instance

    def init_graph(self, save: bool = False):
        """Initialize the entity graph"""
        self.logger.info("Initializing entity graph...")
        log_messages = self.plan_graph.init(save=save)
        # self.logger.info("\n".join(log_messages))
        return log_messages
    
    def load_graph(self, entity_graph_path: str, relation_graph_path: str):
        """Load existing graphs from files"""
        self.logger.info(f"Loading graphs from {entity_graph_path} and {relation_graph_path}")
        log_messages = self.plan_graph.load_graphs(entity_graph_path, relation_graph_path)
        # self.logger.info("\n".join(log_messages))
        return log_messages
    
    def init(self):
        """Initialize the conversation and return the first AI message"""
        log_messages = []
        self.logger.info("Initializing conversation...")
        # log_messages.append("Initializing conversation...")
        
        hint_message, plan_status, hint_log_messages = self.plan_graph.get_hint_message()
        log_messages.extend(hint_log_messages)
        
        self.messages.append(SystemMessage(content=hint_message))
        
        self.logger.info(f"Initial hint message: {hint_message}")
        # log_messages.append(f"Initial hint message: {hint_message[:100]}...")
        
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        
        processed_response = self._process_response(response.content)
        response_content = processed_response["response"]
        think_content = processed_response["think"]
        
        if think_content:
            self.think_history.append({
                "turn": 0,
                "think": think_content,
            })
            self.logger.debug(f"Think content length: {len(think_content)}")
            # log_messages.append(f"Captured think content of length: {len(think_content)}")

        
        self.messages.pop()  # Remove the hint message
        self.messages.append(AIMessage(content=response_content))
        # Store the entire message history for later reference, the entire conversations do not contain hint messages
        self.entire_messages.append(AIMessage(content=response_content))
        self.current_hint = hint_message
        
        self.logger.info(f"Conversation initialized, initial response length: {len(response_content)}")
        # log_messages.append(f"Conversation initialized, initial response length: {len(response_content)}")
        
        return response_content, log_messages
    
    def _get_message_turns(self) -> int:
        """Get the number of turns in the conversation"""
        return len(self.messages) // 2
    
    def conversation(
        self,
        human_message: str,
        images: Optional[List[str]] = None
    ):
        """
        Process a conversation turn and return AI response.

        Args:
            human_message: User's text message
            images: Optional list of base64-encoded images for analysis

        Returns:
            Tuple of (response_content, accomplish, analysis_report, log_messages)
        """
        log_messages = []
        self.logger.info("Processing conversation turn...")
        # log_messages.append("Processing conversation turn...")

        analysis_report = None

        # If images provided, analyze them first and enhance the message
        if images and self.image_analyzer.is_available():
            human_message, analysis_report, image_logs = self._analyze_images(human_message, images)
            log_messages.extend(image_logs)

        # Continue with normal conversation flow
        query_message = self.messages[-1].content if self.messages else ""

        # Update graph with new information
        update_log_messages = self.plan_graph.accept_message(
            self.current_hint, query_message, human_message, is_image_report=images is not None and len(images) > 0
        )
        log_messages.extend(update_log_messages)
        
        # Get new hint
        self.logger.info("Getting new hint message...")
        # log_messages.append("Getting new hint message...")
        hint_message, plan_status, hint_log_messages = self.plan_graph.get_hint_message()
        log_messages.extend(hint_log_messages)
        
        self.current_hint = hint_message
        
        # Prepare messages for this turn
        # reserve the last messages according to the message_reserve_turns
        if self._get_message_turns() > self.message_reserve_turns:
            original_len = len(self.messages)
            self.messages = self.messages[-(self.message_reserve_turns * 2):]
            self.logger.info(f"Trimmed message history from {original_len} to {len(self.messages)}")
            # log_messages.append(f"Trimmed message history from {original_len} to {len(self.messages)}")

        self.messages.append(HumanMessage(content=human_message))
        self.messages.append(SystemMessage(content=hint_message))
        # Store the human message to the entire conversation messages for later reference
        self.entire_messages.append(HumanMessage(content=human_message))
        
        self.logger.info(f"Conversation turn with hint: {hint_message[:100]}...")
        # log_messages.append(f"Conversation turn with hint: {hint_message[:100]}...")
        
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        
        # Process response
        turn_index = len(self.think_history) + 1
        processed_response = self._process_response(response.content)
        response_content = processed_response["response"]
        think_content = processed_response["think"]
        
        if think_content:
            self.think_history.append({
                "turn": turn_index,
                "think": think_content,
            })
            self.logger.debug(f"Think content length: {len(think_content)}")
            # log_messages.append(f"Captured think content of length: {len(think_content)}")

        self.messages.pop()  # Remove hint message
        self.messages.append(AIMessage(content=response_content))
        self.entire_messages.append(AIMessage(content=response_content))

        self.logger.info(f"Conversation turn completed, response length: {len(response_content)}")
        # log_messages.append(f"Conversation turn completed, response length: {len(response_content)}")

        if self.plan_graph.accomplish:
            self.logger.info("Conversation goal accomplished!")
            # log_messages.append("Conversation goal accomplished!")

        return response_content, self.plan_graph.accomplish, analysis_report, log_messages


    def _analyze_images(self, human_message: str, images: List[str]) -> Tuple[str, Optional[Dict[str, Any]], List[str]]:
        """
        Analyze images and enhance human message with analysis report.

        This method uses a TWO-STEP process:
        1. Quick classification: Identify image type without graph context
        2. Detailed analysis: Use type-specific prompt with graph context

        Args:
            human_message: User's text message
            images: List of base64-encoded images

        Returns:
            Tuple of (enhanced_message, analysis_report_dict, log_messages)
        """
        log_messages = []
        self.logger.info(f"Processing {len(images)} image(s) with VLM...")

        # ============================================================
        # STEP 1: Quick Classification (no graph context)
        # ============================================================
        self.logger.info("Step 1: Quick classification...")
        log_messages.append("Step 1: Identifying image type...")

        try:
            classification = self.image_analyzer.quick_classify(
                images=images,
                image_type="base64",
                user_message=human_message
            )
            image_type = classification.get("image_type", "Other Medical Image")
            brief_content = classification.get("brief_content", "")
            confidence = classification.get("confidence", 0.0)

            self.logger.info(f"Classified as: {image_type} (confidence: {confidence})")
            log_messages.append(f"Image type: {image_type}")
            log_messages.append(f"Brief: {brief_content}")

        except Exception as e:
            error_msg = f"Quick classification failed: {e}"
            self.logger.error(error_msg)
            log_messages.append(error_msg)
            # Fallback to generic type
            image_type = "Other Medical Image"
            brief_content = "Classification failed, using generic analysis"
            confidence = 0.0

        # ============================================================
        # STEP 2: Detailed Analysis (with graph context and type-specific prompt)
        # ============================================================
        self.logger.info("Step 2: Detailed analysis...")
        log_messages.append("Step 2: Extracting detailed information...")

        # Get graph context for detailed analysis
        graph_context = self.plan_graph._serialize_nodes_with_value(self.plan_graph.entity_graph)
        self.logger.info(f"Retrieved graph context: {len(graph_context)} chars")

        # Determine which prompt to use based on image type
        prompt_map = {
            "Laboratory Report": "IMAGE_ANALYSIS_LAB_REPORT",
            "ECG": "IMAGE_ANALYSIS_ECG",
            "X-ray": "IMAGE_ANALYSIS_IMAGING",
            "CT Scan": "IMAGE_ANALYSIS_IMAGING",
            "MRI": "IMAGE_ANALYSIS_IMAGING",
            "Ultrasound": "IMAGE_ANALYSIS_IMAGING",
            "Pathology Report": "IMAGE_ANALYSIS_LAB_REPORT",
        }

        prompt_key = prompt_map.get(image_type, "IMAGE_ANALYSIS_GENERIC")

        # Build type-specific analysis query
        analysis_query = self.conv_prompts.get(
            prompt_key,
            target=self.target,
            language=self.plan_graph.language,
            graph_context=graph_context if graph_context.strip() else "No information collected yet.",
            user_message=human_message
        )

        # Analyze images using VLM with type-specific prompt
        try:
            analysis_report, image_logs = self.image_analyzer.analyze(
                query=analysis_query,
                images=images,
                image_type="base64"
            )
            log_messages.extend(image_logs)
            self.logger.info(f"Received detailed analysis report ({len(analysis_report)} chars)")
            log_messages.append("Detailed analysis complete")
        except Exception as e:
            error_msg = f"Detailed analysis failed: {e}"
            self.logger.error(error_msg)
            log_messages.append(error_msg)
            # Return original message and None report if analysis fails
            return human_message, None, log_messages

        # ============================================================
        # Create structured analysis report with classification info
        # ============================================================
        analysis_report_dict = {
            "classification": {
                "image_type": image_type,
                "brief_content": brief_content,
                "confidence": confidence
            },
            "findings": [
                f"影像类型 / Image Type: {image_type}",
                f"简要描述 / Brief: {brief_content}",
                "AI 正在生成详细分析 / AI is generating detailed analysis",
                analysis_report[:500] + "..." if len(analysis_report) > 500 else analysis_report
            ],
            "full_report": analysis_report,
            "recommendation": "AI 正在生成详细分析报告 / AI is generating detailed analysis",
            "image_count": len(images)
        }

        # Enhance human message with classification and analysis report
        if human_message.strip():
            enhanced_message = f"""{human_message}

[Image Classification]
Type: {image_type}
Brief: {brief_content}

[Detailed Analysis Report]
{analysis_report}"""
        else:
            enhanced_message = f"""[Image Classification]
Type: {image_type}
Brief: {brief_content}

[Detailed Analysis Report]
{analysis_report}"""

        self.logger.info("Enhanced message with image analysis report")

        return enhanced_message, analysis_report_dict, log_messages

class GeneralConversation(BaseConversation):
    """General conversation without graph tracking"""
    
    def __init__(self, prompt: str, chat_model, working_directory: Optional[str] = None, stream: bool = False):
        """
        Initialize the general conversation class.
        """
        super().__init__(chat_model)
        self.prompt = prompt
        self.working_directory = working_directory
        self.stream = stream
        
        log_messages = self._ensure_working_directory()
        # self.logger.info("\n".join(log_messages))

    def _ensure_working_directory(self):
        """Ensure working directory exists"""
        log_messages = []
        if self.working_directory and not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
            self.logger.info(f"Created working directory: {self.working_directory}")
            # log_messages.append(f"Created working directory: {self.working_directory}")
        else:
            self.logger.info("Working directory already exists or not specified")
            # log_messages.append("Working directory already exists or not specified")
        return log_messages

    def init(self):
        """Initialize conversation with system prompt"""
        log_messages = []
        self.logger.info("Initializing general conversation...")
        # log_messages.append("Initializing general conversation...")

        self.messages.append(SystemMessage(content=self.prompt))
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        self.messages.append(response)

        self.logger.info(f"Conversation initialized, initial response length: {len(response.content)}")
        # log_messages.append(f"Conversation initialized, initial response length: {len(response.content)}")

        return response.content, log_messages
    
    def conversation(self, human_message: str):
        """Process a conversation turn"""
        log_messages = []
        self.logger.info("Processing conversation turn...")
        # log_messages.append("Processing conversation turn...")

        self.messages.append(HumanMessage(content=human_message))
        response = self.chat_model.invoke(self.messages, stream=self.stream)
        self.messages.append(response)
        
        self.logger.info(f"Conversation turn completed, response length: {len(response.content)}")
        # log_messages.append(f"Conversation turn completed, response length: {len(response.content)}")
        
        return response.content, log_messages