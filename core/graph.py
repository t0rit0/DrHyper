import contextlib
import json
import math
import os
import pickle
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any

import networkx as nx
import numpy as np
from langchain.schema import HumanMessage, SystemMessage

from drhyper.config.settings import ConfigManager
from drhyper.core.schemas import (
    ENTITY_EDGES_SCHEMA,
    ENTITY_NODES_SCHEMA,
    ENTITY_RETRIEVE_SCHEMA,
    EXTRACT_INFO_SCHEMA,
    PRESET_PROPAGATION_SCHEMA,
    UPDATE_GRAPH_SCHEMA,
)
from drhyper.prompts.templates import GraphPrompts
from drhyper.utils.aux import parse_json_response
from drhyper.utils.logging import get_logger


class EntityGraph:
    """Entity graph for tracking conversation information"""

    def __init__(
        self,
        target: str,
        graph_model,
        conv_model,
        routine: str | None = None,
        working_directory: str | None = None,
        language: str = "English",
        **params
    ):
        self.config = ConfigManager()
        self.prompts = GraphPrompts()
        self.target = target
        self.graph_model = graph_model
        self.conv_model = conv_model
        self.routine = routine
        self.working_directory = working_directory
        self.language = language

        # Graph parameters
        self.node_hit_threshold = params.get('node_hit_threshold', np.inf)
        self.confidential_threshold = params.get('confidential_threshold', 0.2)
        self.relevance_threshold = params.get('relevance_threshold', 0.2)
        self.weight_threshold = params.get('weight_threshold', 0.8)
        self.alpha = params.get('alpha', 10.0)
        self.beta = params.get('beta', 1.0)
        self.gamma = params.get('gamma', 1.0)

        self.step = 0
        self.accomplish = False
        self.prev_node = None
        self.key_nodes = []  # Key nodes identified for diagnosis
        self.logger = get_logger(self.__class__.__name__)

        # Initialize empty graphs (will be populated by init() or load_graphs())
        self.entity_graph = nx.DiGraph()
        self.relation_graph = nx.DiGraph()

        self._ensure_working_directory()

    def _ensure_working_directory(self):
        """Ensure working directory exists"""
        if self.working_directory and not os.path.exists(self.working_directory):
            os.makedirs(self.working_directory)
            self.logger.info(f"Created working directory: {self.working_directory}")
            log_messages = [f"Created working directory: {self.working_directory}"]
            return log_messages
        self.logger.info("Working directory already exists")
        log_messages = ["Working directory already exists"]
        return log_messages

    def init(
        self,
        save: bool = False,
        patient_context: dict[str, Any] | None = None,
        patient_id: str | None = None,
        db_session=None,
    ):
        """Initialize the graph

        Args:
            save: Whether to save graphs to disk
            patient_context: Optional patient context with patient text records
            patient_id: Optional patient ID for metric preset injection
            db_session: Optional DB session for metric preset injection
        """
        log_messages = []

        # if patient_context and patient_context.get("known_entities"):
        #     prefill_messages = self._prefill_known_entities(patient_context["known_entities"])
        #     log_messages.extend(prefill_messages)

        patient_text_records = None
        if patient_context and patient_context.get("patient_text_records"):
            patient_text_records = patient_context["patient_text_records"]

        init_messages = self._initialize_graph(
            patient_text_records=patient_text_records,
            patient_id=patient_id,
            db_session=db_session,
        )
        log_messages.extend(init_messages)

        clustering_messages = self._clustering()
        log_messages.extend(clustering_messages)

        if save:
            save_messages = self.save_graphs(self.working_directory)
            log_messages.extend(save_messages)

        return log_messages

    def save_graphs(self, output_dir: str):
        """Save entity and relation graphs to files"""
        log_messages = []
        if not self.working_directory:
            self.logger.info("No working directory specified, graphs not saved")
            # log_messages.append("No working directory specified, graphs not saved")
            return log_messages

        entity_graph_file = os.path.join(output_dir, "entity_graph.pkl")
        relation_graph_file = os.path.join(output_dir, "relation_graph.pkl")

        with open(entity_graph_file, "wb") as f:
            pickle.dump(self.entity_graph, f)
        with open(relation_graph_file, "wb") as f:
            pickle.dump(self.relation_graph, f)

        self.logger.info(f"Saved graphs to {self.working_directory}")
        # log_messages.append(f"Saved graphs to {self.working_directory}")
        return log_messages

    def load_graphs(self, entity_graph_path: str, relation_graph_path: str):
        """Load graphs from files"""
        log_messages = []
        if not os.path.exists(entity_graph_path):
            error_msg = f"Entity graph not found: {entity_graph_path}"
            self.logger.error(error_msg)
            # log_messages.append(error_msg)
            raise FileNotFoundError(error_msg)
        if not os.path.exists(relation_graph_path):
            error_msg = f"Relation graph not found: {relation_graph_path}"
            self.logger.error(error_msg)
            # log_messages.append(error_msg)
            raise FileNotFoundError(error_msg)

        with open(entity_graph_path, "rb") as f:
            self.entity_graph = pickle.load(f)
        with open(relation_graph_path, "rb") as f:
            self.relation_graph = pickle.load(f)

        clustering_messages = self._clustering()
        log_messages.extend(clustering_messages)

        self.logger.info("Loaded graphs successfully")
        # log_messages.append("Loaded graphs successfully")
        return log_messages

    def _initialize_graph(
        self,
        patient_text_records: dict[str, str] | None = None,
        patient_id: str | None = None,
        db_session=None,
    ):
        """Initialize entity and relation graphs using LLM

        Args:
            patient_text_records: Optional patient text records for context during entity initialization
            patient_id: Optional patient ID for metric preset injection
            db_session: Optional DB session for metric preset injection.
                        When both patient_id and db_session are provided,
                        metric preset nodes are injected.
        """
        log_messages = []

        # Step 1: Retrieve entities
        # self.logger.info("Retrieving entities...")
        entities, entity_messages = self._retrieve_entities()
        log_messages.extend(entity_messages)

        # Step 2: Initialize entity attributes
        # self.logger.info("Initializing entity attributes...")
        nodes, node_messages = self._initialize_entity_attributes(entities, patient_text_records)
        log_messages.extend(node_messages)

        # Step 3: Inject metric presets (no LLM, direct DB queries)
        preset_nodes = []
        if patient_id and db_session:
            try:
                from backend.services.metric_presets import inject_metric_presets
                preset_nodes = inject_metric_presets(
                    patient_id=patient_id,
                    db=db_session,
                )
                self.logger.info(f"Injected {len(preset_nodes)} metric preset nodes")
            except Exception as e:
                self.logger.warning(f"Metric preset injection failed (non-fatal): {e}")

        # Step 4 & 5: Create edges in parallel
        # Entity edges: only original entities (presets are data, not diagnostic dependencies)
        # Relation edges: original entities + presets (presets should be semantically linked)
        with ThreadPoolExecutor(max_workers=2) as executor:
            entity_future = executor.submit(self._create_entity_edges, entities)

            relation_entities = list(entities)
            if preset_nodes:
                relation_entities.extend(
                    [{"id": p["id"], "name": p["name"]} for p in preset_nodes]
                )
            relation_future = executor.submit(self._create_relation_edges, relation_entities)

            entity_edges, entity_edge_messages = entity_future.result()
            relation_edges, relation_edge_messages = relation_future.result()

        log_messages.extend(entity_edge_messages)
        log_messages.extend(relation_edge_messages)

        # Build graphs (original nodes + preset nodes)
        all_nodes = nodes + preset_nodes
        has_prefilled_nodes = self.entity_graph.number_of_nodes() > 0
        has_new_nodes = len(all_nodes) > 0

        if has_prefilled_nodes and not has_new_nodes:
            self.logger.info("Preserving prefilled nodes, no new entities to add")
            entity_graph_messages = []
            relation_graph_messages = []
        elif has_prefilled_nodes and has_new_nodes:
            entity_graph_messages = self._add_nodes_to_existing_graph(self.entity_graph, all_nodes, entity_edges)
            log_messages.extend(entity_graph_messages)

            relation_graph_messages = self._add_nodes_to_existing_graph(self.relation_graph, all_nodes, relation_edges)
            log_messages.extend(relation_graph_messages)
        else:
            self.entity_graph, entity_graph_messages = self._build_graph(all_nodes, entity_edges)
            log_messages.extend(entity_graph_messages)

            self.relation_graph, relation_graph_messages = self._build_graph(all_nodes, relation_edges)
            log_messages.extend(relation_graph_messages)

        # Initialize node states
        node_states_messages = self._initialize_node_states()
        log_messages.extend(node_states_messages)

        # Propagate preset values to entity nodes, then clean up
        if preset_nodes:
            propagation_messages = self._propagate_preset_values(preset_nodes)
            log_messages.extend(propagation_messages)

            removal_messages = self._remove_preset_nodes_from_entity_graph(preset_nodes)
            log_messages.extend(removal_messages)

            cleanup_messages = self._cleanup_unlinked_presets(preset_nodes)
            log_messages.extend(cleanup_messages)

        return log_messages


    def _retrieve_entities(self) -> tuple[list[dict[str, str]], list[str]]:
        """Retrieve entities needed for the target"""
        messages = []
        entities = []
        log_messages = []

        prompt = self.prompts.get("ENTITY_RETRIEVE", purpose=self.target, language=self.language)
        if self.routine:
            routine_prompt = self.prompts.get("ROUTINE_ADDITION", routine=self.routine)
            prompt += "\n" + routine_prompt

        messages.append(SystemMessage(content=prompt))
        response = self.graph_model.invoke(messages, response_format=ENTITY_RETRIEVE_SCHEMA)

        try:
            result = parse_json_response(response.content)
            entities.extend(result.get("entities", []))
            messages.append(response)

            # Continue if needed
            endpoint = result.get("endpoint", True)
            if isinstance(endpoint, str):
                endpoint = endpoint.lower() == "true"
            elif isinstance(endpoint, bool):
                endpoint = endpoint
            else:
                raise ValueError(f"Unexpected endpoint type: {type(endpoint)}")
            iteration = 1

            while not endpoint and iteration < 10:
                messages.append(HumanMessage(content=self.prompts.get("CONTINUE_ENTITY_RETRIEVE")))
                response = self.graph_model.invoke(messages, response_format=ENTITY_RETRIEVE_SCHEMA)
                result = parse_json_response(response.content)

                new_entities = result.get("entities", [])
                if not new_entities:
                    break

                entities.extend(new_entities)
                messages.append(response)
                endpoint = result.get("endpoint", True)
                if isinstance(endpoint, str):
                    endpoint = endpoint.lower() == "true"
                elif isinstance(endpoint, bool):
                    endpoint = endpoint
                else:
                    raise ValueError(f"Unexpected endpoint type: {type(endpoint)}")
                iteration += 1

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse entity response: {e}"
            self.logger.error(f"response.content: {response.content}")
            self.logger.error(error_msg)
            # log_messages.append(error_msg)
            raise

        # Assign IDs to entities
        entities_with_ids = [{"id": f"v{i}", "name": entity} for i, entity in enumerate(entities, start=1)]
        self.logger.info(f"Retrieved {len(entities_with_ids)} entities")
        # log_messages.append(f"Retrieved {len(entities_with_ids)} entities")

        return entities_with_ids, log_messages

    def _initialize_chunk_attributes(
        self, chunk: list[dict[str, str]], chunk_index: int, total_chunks: int,
        patient_context_str: str
    ) -> list[dict[str, Any]]:
        """Initialize attributes for a single chunk of entities.

        Args:
            chunk: List of entities in this chunk
            chunk_index: 0-based chunk index
            total_chunks: Total number of chunks
            patient_context_str: Formatted patient context string

        Returns:
            List of entity nodes with initialized attributes
        """
        entities_str = ", ".join([f"id: {e['id']}, name: {e['name']}" for e in chunk])
        prompt = self.prompts.get("INIT_GRAPH_ENTITY", purpose=self.target, entities=entities_str, language=self.language)
        if patient_context_str:
            prompt = f"{patient_context_str}\n\n{prompt}"

        response = self.graph_model.invoke([HumanMessage(content=prompt)], response_format=ENTITY_NODES_SCHEMA)
        result = parse_json_response(response.content)
        chunk_nodes = result.get("entities", [])
        self.logger.info(f"Initialized attributes for chunk {chunk_index + 1}/{total_chunks}")
        return chunk_nodes

    @staticmethod
    def _ensure_numeric_fields(node: dict[str, Any]) -> dict[str, Any]:
        """Convert numeric fields from LLM JSON responses to proper Python types.

        LLM responses may return numeric values as strings (e.g. "0.8" instead of 0.8).
        This ensures consistency and prevents numpy/math errors downstream.
        """
        float_fields = ("weight", "uncertainty", "confidential_level", "relevance")
        int_fields = ("hit", "community", "status")

        for field in float_fields:
            if field in node:
                with contextlib.suppress(ValueError, TypeError):
                    node[field] = float(node[field])

        for field in int_fields:
            if field in node:
                with contextlib.suppress(ValueError, TypeError):
                    node[field] = int(node[field])

        return node

    def _initialize_entity_attributes(
        self, entities: list[dict[str, str]], patient_text_records: dict[str, str] | None = None
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """Initialize attributes for entities

        Args:
            entities: List of entities to initialize
            patient_text_records: Optional patient text records for context
        """
        log_messages = []
        chunk_size = 5

        # 格式化患者文本记录为上下文字符串
        patient_context_str = ""
        if patient_text_records:
            patient_context_str = self._format_patient_text_records(patient_text_records)

        # Build chunks
        chunks = []
        for i in range(0, len(entities), chunk_size):
            chunks.append(entities[i:i + chunk_size])

        total_chunks = len(chunks)

        # Parallelize chunk processing
        chunk_results: dict[int, list[dict[str, Any]]] = {}
        with ThreadPoolExecutor(max_workers=min(total_chunks, 4)) as executor:
            futures = {
                executor.submit(
                    self._initialize_chunk_attributes,
                    chunk, idx, total_chunks, patient_context_str
                ): idx
                for idx, chunk in enumerate(chunks)
            }
            for future in as_completed(futures):
                idx = futures[future]
                chunk_results[idx] = future.result()

        # Merge results in original order
        nodes = []
        for idx in sorted(chunk_results.keys()):
            nodes.extend(chunk_results[idx])

        # 为每个节点添加时间属性
        now = datetime.now()
        for node in nodes:
            self._ensure_numeric_fields(node)
            node["extracted_at"] = now
            node["last_updated_at"] = now
            node["source"] = "conversation"
            node["original_confidential_level"] = node.get("confidential_level", 0.5)

        self.logger.info(f"Total number of nodes: {len(nodes)}")
        return nodes, log_messages

    def _total_node_number(self) -> int:
        """Get total number of nodes in the entity graph"""
        count = self.entity_graph.number_of_nodes()
        # self.logger.info(f"Total node count: {count}")
        return count

    def _accomplished_node_number(self) -> int:
        """Get number of nodes with status 2 (accomplished)"""
        count = sum(1 for node in self.entity_graph.nodes(data=True) if node[1].get("status") == 2)
        # self.logger.info(f"Accomplished node count: {count}")
        return count

    def _remaining_node_number(self) -> int:
        """Get number of nodes with status 0 or 1 (not accomplished)"""
        count = sum(1 for node in self.entity_graph.nodes(data=True) if node[1].get("status") in (0, 1))
        # self.logger.info(f"Remaining node count: {count}")
        return count

    def _create_entity_edges(self, entities: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
        """Create edges for entity graph (dependencies)"""
        edges, log_messages = self._create_edges(entities, "INIT_ENTITY_GRAPH_EDGES", "CONTINUE_INIT_ENTITY_GRAPH_EDGES")
        self.logger.info(f"Created {len(edges)} entity graph edges")
        # log_messages.append(f"Created {len(edges)} entity graph edges")
        return edges, log_messages

    def _create_relation_edges(self, entities: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[str]]:
        """Create edges for relation graph"""
        edges, log_messages = self._create_edges(entities, "INIT_RELATION_GRAPH_EDGES", "CONTINUE_INIT_RELATION_GRAPH_EDGES")
        self.logger.info(f"Created {len(edges)} relation graph edges")
        # log_messages.append(f"Created {len(edges)} relation graph edges")
        return edges, log_messages

    def _create_edges(self, entities: list[dict[str, str]], init_prompt_key: str, continue_prompt_key: str) -> tuple[list[dict[str, str]], list[str]]:
        """Generic edge creation method"""
        messages = []
        edges = []
        log_messages = []
        iteration = 0
        endpoint = False
        entities_str = ", ".join([f"id: {e['id']}, name: {e['name']}" for e in entities])

        while not endpoint and iteration < 10:
            if iteration == 0:
                prompt = self.prompts.get(init_prompt_key, purpose=self.target, entities=entities_str, language=self.language)
            else:
                prompt = self.prompts.get(continue_prompt_key)

            messages.append(HumanMessage(content=prompt))
            response = self.graph_model.invoke(messages, response_format=ENTITY_EDGES_SCHEMA)

            try:
                result = parse_json_response(response.content)
                new_edges = result.get("edges", [])

                if not new_edges:
                    self.logger.info(f"No new edges in iteration {iteration+1}")
                    # log_messages.append(f"No new edges in iteration {iteration+1}")
                    break

                edges.extend(new_edges)
                self.logger.info(f"Added {len(new_edges)} edges in iteration {iteration+1}")
                # log_messages.append(f"Added {len(new_edges)} edges in iteration {iteration+1}")

                endpoint = result.get("endpoint", True)
                if isinstance(endpoint, str):
                    endpoint = endpoint.lower() == "true"
                elif isinstance(endpoint, bool):
                    endpoint = endpoint
                else:
                    error_msg = f"Unexpected endpoint type: {type(endpoint)}"
                    self.logger.error(error_msg)
                    # log_messages.append(error_msg)
                    raise ValueError(error_msg)

                messages.append(response)
                iteration += 1

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse edges: {e}"
                self.logger.error(error_msg)
                # log_messages.append(error_msg)
                break

        return edges, log_messages

    def _build_graph(self, nodes: list[dict[str, Any]], edges: list[dict[str, str]]) -> tuple[nx.DiGraph, list[str]]:
        """Build NetworkX graph from nodes and edges"""
        G = nx.DiGraph()
        log_messages = []

        node_count = 0
        for node in nodes:
            node_id = node.get("id")
            if node_id:
                self._ensure_numeric_fields(node)
                G.add_node(node_id, **node)
                node_count += 1

        edge_count = 0
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target and source in G and target in G:
                G.add_edge(source, target, **edge)
                edge_count += 1

        self.logger.info(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        # log_messages.append(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        # log_messages.append(f"Added {node_count} nodes and {edge_count} edges to graph")

        return G, log_messages

    def _add_nodes_to_existing_graph(
        self, graph: nx.DiGraph, nodes: list[dict[str, Any]], edges: list[dict[str, str]]
    ) -> list[str]:
        """添加节点和边到现有图中（用于保留预填充节点）"""
        log_messages = []

        # 添加新节点
        node_count = 0
        for node in nodes:
            node_id = node.get("id")
            if node_id and node_id not in graph:
                self._ensure_numeric_fields(node)
                graph.add_node(node_id, **node)
                node_count += 1

        # 添加边
        edge_count = 0
        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")
            if source and target and source in graph and target in graph:
                graph.add_edge(source, target, **edge)
                edge_count += 1

        self.logger.info(f"Added {node_count} nodes and {edge_count} edges to existing graph")
        return log_messages

    def _initialize_node_states(self):
        """Initialize node states (value, hit, status)"""
        log_messages = []
        initialized_count = 0

        for node in self.entity_graph.nodes:
            node_updated = False

            if "value" not in self.entity_graph.nodes[node]:
                self.entity_graph.nodes[node]["value"] = ""
                node_updated = True

            if "hit" not in self.entity_graph.nodes[node]:
                self.entity_graph.nodes[node]["hit"] = 0
                node_updated = True

            if "status" not in self.entity_graph.nodes[node]:
                self.entity_graph.nodes[node]["status"] = 0
                node_updated = True

            if node_updated:
                initialized_count += 1

        self.logger.info(f"Initialized states for {initialized_count} nodes")
        # log_messages.append(f"Initialized states for {initialized_count} nodes")
        return log_messages

    def _cleanup_unlinked_presets(self, preset_nodes: list[dict[str, Any]]) -> list[str]:
        """
        Remove preset nodes that have no edges in the relation graph.

        Presets with no relation edges are orphaned data nodes that
        provide no diagnostic value. Removing them keeps the graph clean.
        """
        log_messages = []
        removed_count = 0

        for preset_node in preset_nodes:
            node_id = preset_node["id"]

            if not self.relation_graph.has_node(node_id):
                continue

            has_edges = (
                len(list(self.relation_graph.predecessors(node_id))) > 0
                or len(list(self.relation_graph.successors(node_id))) > 0
            )

            if not has_edges:
                if self.entity_graph.has_node(node_id):
                    self.entity_graph.remove_node(node_id)
                if self.relation_graph.has_node(node_id):
                    self.relation_graph.remove_node(node_id)
                removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Cleaned up {removed_count} unlinked preset nodes")

        return log_messages

    def _propagate_preset_values(self, preset_nodes: list[dict[str, Any]]) -> list[str]:
        """
        Propagate preset metric values to connected entity nodes via LLM.

        Finds entity nodes connected to presets in relation_graph, collects
        the preset data for each entity, and asks the LLM to decide what
        value/confidence/weight to write to each entity node.

        Each entity is processed in parallel using ThreadPoolExecutor.

        Returns:
            List of log messages.
        """
        log_messages = []

        # Step 1: Find preset nodes that have edges in relation_graph
        linked_preset_ids = set()
        for preset_node in preset_nodes:
            node_id = preset_node["id"]
            if (self.relation_graph.has_node(node_id)
                    and (len(list(self.relation_graph.predecessors(node_id))) > 0
                         or len(list(self.relation_graph.successors(node_id))) > 0)):
                    linked_preset_ids.add(node_id)

        if not linked_preset_ids:
            self.logger.info("No linked preset nodes found for propagation")
            return log_messages

        # Step 2: Find entity nodes connected to presets in relation_graph
        entity_ids_with_presets = set()
        for preset_id in linked_preset_ids:
            neighbors = set(self.relation_graph.predecessors(preset_id))
            neighbors.update(self.relation_graph.successors(preset_id))
            for neighbor_id in neighbors:
                if (neighbor_id in self.entity_graph
                        and self.entity_graph.nodes[neighbor_id].get("source") != "preset"):
                    entity_ids_with_presets.add(neighbor_id)

        if not entity_ids_with_presets:
            self.logger.info("No entity nodes connected to presets for propagation")
            return log_messages

        self.logger.info(
            f"Found {len(entity_ids_with_presets)} entity nodes connected to "
            f"{len(linked_preset_ids)} preset nodes for propagation"
        )

        # Step 3: For each entity node, collect connected preset data
        entity_preset_map: dict[str, list[dict[str, Any]]] = {}
        for entity_id in entity_ids_with_presets:
            entity_preset_map[entity_id] = []
            seen_preset_ids = set()
            for neighbor_id in list(self.relation_graph.predecessors(entity_id)) + list(self.relation_graph.successors(entity_id)):
                if neighbor_id in linked_preset_ids and neighbor_id not in seen_preset_ids:
                    seen_preset_ids.add(neighbor_id)
                    node_data = self.relation_graph.nodes[neighbor_id]
                    entity_preset_map[entity_id].append({
                        "id": neighbor_id,
                        "name": node_data.get("name", ""),
                        "description": node_data.get("description", ""),
                        "value": node_data.get("value", ""),
                        "metric_name": node_data.get("metric_name", ""),
                    })

        # Step 4: Process each entity in parallel
        entity_ids_list = sorted(entity_ids_with_presets)

        with ThreadPoolExecutor(max_workers=min(len(entity_ids_list), 4)) as executor:
            futures = {
                executor.submit(
                    self._propagate_single_entity,
                    eid,
                    entity_preset_map.get(eid, []),
                ): eid
                for eid in entity_ids_list
            }

            for future in as_completed(futures):
                eid = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.error(f"Failed to propagate to entity {eid}: {e}")

        # Count updated nodes
        total_updated = sum(
            1 for nid in entity_ids_with_presets
            if self.entity_graph.nodes[nid].get("value")
            and self.entity_graph.nodes[nid].get("source") != "preset"
        )
        self.logger.info(f"Preset propagation complete: {total_updated}/{len(entity_ids_with_presets)} entity nodes updated")
        return log_messages

    def _propagate_single_entity(
        self,
        entity_id: str,
        connected_presets: list[dict[str, Any]],
    ) -> None:
        """
        Process a single entity node: call LLM to decide value from presets,
        then apply the update to entity_graph.

        Args:
            entity_id: The entity node ID.
            connected_presets: List of preset node data dicts connected to this entity.
        """
        entity_data = self.entity_graph.nodes[entity_id]

        # Format entity info
        entity_nodes_str = (
            f"node {entity_id}, {entity_data['name']}, "
            f"description: {entity_data.get('description', '')}, "
            f"current weight: {entity_data.get('weight', 0)}, "
            f"current uncertainty: {entity_data.get('uncertainty', 1.0)}"
        )

        # Format connected preset data
        if connected_presets:
            preset_info = "; ".join([
                f"{p['name']}: {p['value']} (metric: {p['metric_name']})"
                for p in connected_presets
            ])
            preset_data_str = f"  Entity {entity_id} ({entity_data['name']}): [{preset_info}]"
        else:
            preset_data_str = f"  Entity {entity_id}: no preset data connected"

        prompt = self.prompts.get(
            "PRESET_PROPAGATION",
            purpose=self.target,
            entity_nodes=entity_nodes_str,
            preset_data=preset_data_str,
        )

        response = self.graph_model.invoke(
            [SystemMessage(content=prompt)],
            response_format=PRESET_PROPAGATION_SCHEMA,
        )

        try:
            updates = parse_json_response(response.content)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse propagation response for {entity_id}: {e}")
            return

        for update in updates:
            node_id = update.get("id")
            value = str(update.get("value", "")).strip()
            try:
                confidential_level = float(update.get("confidential_level", 0.0))
            except (ValueError, TypeError):
                confidential_level = 0.0

            if node_id not in self.entity_graph.nodes:
                self.logger.warning(f"Propagation: node {node_id} not in entity_graph, skipping")
                continue

            if not value or confidential_level <= 0:
                continue

            self.entity_graph.nodes[node_id]["value"] = value
            self.entity_graph.nodes[node_id]["confidential_level"] = confidential_level
            self.entity_graph.nodes[node_id]["last_updated_at"] = datetime.now()

            if "original_confidential_level" not in self.entity_graph.nodes[node_id]:
                self.entity_graph.nodes[node_id]["original_confidential_level"] = confidential_level
            else:
                try:
                    stored = float(self.entity_graph.nodes[node_id]["original_confidential_level"])
                except (ValueError, TypeError):
                    stored = 0.0
                if confidential_level > stored:
                    self.entity_graph.nodes[node_id]["original_confidential_level"] = confidential_level

            if confidential_level >= self.confidential_threshold:
                self.entity_graph.nodes[node_id]["status"] = 2
            else:
                self.entity_graph.nodes[node_id]["status"] = 1

            weight = update.get("weight")
            if weight is not None:
                with contextlib.suppress(ValueError, TypeError):
                    self.entity_graph.nodes[node_id]["weight"] = float(weight)

            uncertainty = update.get("uncertainty")
            if uncertainty is not None:
                with contextlib.suppress(ValueError, TypeError):
                    self.entity_graph.nodes[node_id]["uncertainty"] = float(uncertainty)

            self.logger.info(
                f"Propagated to node {node_id}: value={value[:50]}, "
                f"cl={confidential_level:.2f}, reason={update.get('update_reason', 'N/A')}"
            )

    def _remove_preset_nodes_from_entity_graph(self, preset_nodes: list[dict[str, Any]]) -> list[str]:
        """
        Remove all preset nodes from entity_graph after value propagation.

        Preset nodes remain in relation_graph for PageRank/community detection,
        but are removed from entity_graph so they don't appear in reports or
        hint selection.
        """
        log_messages = []
        removed_count = 0

        for preset_node in preset_nodes:
            node_id = preset_node["id"]
            if self.entity_graph.has_node(node_id):
                self.entity_graph.remove_node(node_id)
                removed_count += 1

        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} preset nodes from entity_graph (kept in relation_graph)")

        return log_messages

    def _update_weights_from_presets(self, preset_nodes: list[dict[str, Any]]) -> list[str]:
        """
        Update weights/uncertainty of neighbors of linked preset nodes.

        Uses the existing LLM-based _update_existing_node_weights() for
        accurate weight recalculation based on preset values.
        """
        log_messages = []
        active_ids = []

        for preset_node in preset_nodes:
            node_id = preset_node["id"]
            if self.relation_graph.has_node(node_id):
                has_edges = (
                    len(list(self.relation_graph.predecessors(node_id))) > 0
                    or len(list(self.relation_graph.successors(node_id))) > 0
                )
                if has_edges and preset_node.get("status", 0) > 0:
                    active_ids.append(node_id)

        if active_ids:
            self.logger.info(f"Updating neighbor weights for {len(active_ids)} linked preset nodes")
            weight_messages = self._update_existing_node_weights(active_ids)
            log_messages.extend(weight_messages)

        return log_messages

    def _clustering(self):
        """Perform community detection on the graph"""
        log_messages = []

        try:
            import igraph as ig
            import leidenalg
            self.logger.info("Using leidenalg/igraph for community detection")
            # log_messages.append("Using leidenalg/igraph for community detection")
        except ImportError:
            self.logger.warning("leidenalg/igraph not installed; assigning all nodes to community 0")
            # log_messages.append("leidenalg/igraph not installed; assigning all nodes to community 0")

            node_count = 0
            for node in self.relation_graph.nodes():
                self.relation_graph.nodes[node]["community"] = 0
                if node in self.entity_graph:
                    self.entity_graph.nodes[node]["community"] = 0
                node_count += 1

            self.logger.info(f"Assigned {node_count} nodes to default community 0")
            # log_messages.append(f"Assigned {node_count} nodes to default community 0")
            return log_messages

        # Convert to undirected graph for community detection
        ud_graph = self.relation_graph.to_undirected()
        mapping = dict(enumerate(ud_graph.nodes()))
        inv_mapping = {v: k for k, v in mapping.items()}

        # Check for edge weights
        has_weights = any('weight' in ud_graph.get_edge_data(u, v, {}) for u, v in ud_graph.edges())

        # Build igraph
        edges = [(inv_mapping[u], inv_mapping[v]) for u, v in ud_graph.edges()]
        ig_g = ig.Graph(len(mapping), edges)
        ig_g.vs["name"] = [mapping[i] for i in range(len(mapping))]

        if has_weights:
            weights = [ud_graph.get_edge_data(u, v).get('weight', 1.0) for u, v in ud_graph.edges()]
            ig_g.es['weight'] = weights
            partition = leidenalg.find_partition(ig_g, leidenalg.RBConfigurationVertexPartition, weights='weight')
            self.logger.info("Using weighted community detection")
            # log_messages.append("Using weighted community detection")
        else:
            partition = leidenalg.find_partition(ig_g, leidenalg.RBConfigurationVertexPartition)
            self.logger.info("Using unweighted community detection")
            # log_messages.append("Using unweighted community detection")

        # Assign communities
        community_counts = {}
        for comm_id, community in enumerate(partition):
            for vid in community:
                node_name = ig_g.vs[vid]["name"]
                self.relation_graph.nodes[node_name]["community"] = comm_id
                if node_name in self.entity_graph:
                    self.entity_graph.nodes[node_name]["community"] = comm_id

                if comm_id not in community_counts:
                    community_counts[comm_id] = 0
                community_counts[comm_id] += 1

        self.logger.info(f"Community detection completed with {len(partition)} communities")
        # log_messages.append(f"Community detection completed with {len(partition)} communities")
        return log_messages

    def get_hint_message(self) -> tuple[str, bool, list[str]]:
        """Generate hint message for next conversation turn"""
        log_messages = []
        selection = self._select_node()
        selection_info = selection[2] if selection else []
        log_messages.extend(selection_info)

        if selection is None or selection[0] is None:
            # All information collected, generate final hint
            self.logger.info("All nodes processed, generating accomplishment hint")
            # log_messages.append("All nodes processed, generating accomplishment hint")

            # Identify key nodes before generating accomplishment hint
            self.logger.info("Identifying key diagnostic nodes...")
            try:
                from drhyper.core.key_node import KeyNodeIdentifier
                key_node_identifier = KeyNodeIdentifier(
                    self.entity_graph,
                    self.relation_graph,
                    self.config
                )
                self.key_nodes = key_node_identifier.identify(
                    percentile_threshold=0.80,
                    min_combined_score=0.60
                )
                self.logger.info(f"Identified {len(self.key_nodes)} key diagnostic nodes")

                # Format key nodes for prompt injection
                key_nodes_info = self._format_key_nodes_for_prompt(self.key_nodes)
                self.logger.info(f"Formatted {len(self.key_nodes)} key nodes for prompt")
            except Exception as e:
                self.logger.error(f"Key node identification failed: {e}")
                self.key_nodes = []
                key_nodes_info = "No key diagnostic findings identified."

            prompt = self.prompts.get(
                "HINT_MESSAGE_ACCOMPLISH",
                collected=self._serialize_nodes_with_value(self.entity_graph),
                purpose=self.target,
                language=self.language,
                key_nodes_info=key_nodes_info
            )
            response = self.graph_model.invoke([SystemMessage(content=prompt)])
            hint_message = response.content
            self.accomplish = True

            self.logger.info("Generated accomplishment hint")
            # log_messages.append("Generated accomplishment hint")
        else:
            # Generate hint for next node
            node_id, node_data = selection[0], selection[1]
            self.logger.info(f"Generating hint for node {node_id}: {node_data.get('name', '')}")
            # log_messages.append(f"Generating hint for node {node_id}: {node_data.get('name', '')}")

            prompt = self.prompts.get(
                "HINT_MESSAGE_RETRIEVE",
                collected=self._serialize_nodes_with_value(self.entity_graph),
                recommendation=self._serialize_node_info(node_data),
                purpose=self.target,
                language=self.language
            )
            response = self.graph_model.invoke([SystemMessage(content=prompt)])
            hint_message = response.content

            self.logger.info(f"Generated hint for node {node_id}")
            # log_messages.append(f"Generated hint for node {node_id}")

        total_nodes = self._total_node_number()
        accomplished_nodes = self._accomplished_node_number()
        remaining_nodes = self._remaining_node_number()

        self.logger.info(f"Total number of nodes in the graph: {total_nodes}")
        self.logger.info(f"Accomplished number of nodes: {accomplished_nodes}")
        self.logger.info(f"Remaining number of nodes: {remaining_nodes}")
        # log_messages.append(f"Total number of nodes in the graph: {total_nodes}")
        # log_messages.append(f"Accomplished number of nodes: {accomplished_nodes}")
        # log_messages.append(f"Remaining number of nodes: {remaining_nodes}")

        return hint_message, self.accomplish, log_messages

    def accept_message(
        self,
        hint_message: str,
        query_message: str,
        user_message: str,
        is_image_report: bool = False
    ):
        """
        Process user message and update graph.

        Args:
            hint_message: Hint message for context
            query_message: Query message from AI
            user_message: User's response or image analysis report
            is_image_report: If True, treat user_message as image analysis report

        Returns:
            List of log messages
        """
        log_messages = []

        if is_image_report:
            self.logger.info("Processing image analysis report...")
            # log_messages.append("Processing image analysis report...")
        else:
            self.logger.info("Processing user message...")
            # log_messages.append("Processing user message...")

        updated_nodes, new_nodes, extract_messages = self._process_user_message(
            hint_message, query_message, user_message, is_image_report=is_image_report
        )
        log_messages.extend(extract_messages)

        self.logger.info(f"Updated {len(updated_nodes)} existing nodes and added {len(new_nodes)} new nodes")
        # log_messages.append(f"Updated {len(updated_nodes)} existing nodes and added {len(new_nodes)} new nodes")

        update_messages = self._update_graph(updated_nodes, new_nodes)
        log_messages.extend(update_messages)

        # Re-cluster if significant changes from image report
        if is_image_report and (len(new_nodes) > 0 or len(updated_nodes) > 3):
            self.logger.info("Significant graph changes from image report, re-clustering...")
            # log_messages.append("Significant graph changes from image report, re-clustering...")
            clustering_messages = self._clustering()
            log_messages.extend(clustering_messages)

        return log_messages

    def _get_available_nodes(self) -> list[tuple[str, dict[str, Any]]]:
        """Get nodes available for querying"""
        available = []
        log_messages = []

        status_filtered = 0
        hit_filtered = 0
        weight_filtered = 0
        prereq_filtered = 0
        preset_filtered = 0

        for node_id, data in self.entity_graph.nodes(data=True):
            # Exclude preset nodes from hint selection
            if data.get("source") == "preset":
                preset_filtered += 1
                continue

            # Check status
            if data.get("status", 0) not in (0, 1):
                status_filtered += 1
                continue

            # Check hit threshold
            if data.get("hit", 0) >= self.node_hit_threshold:
                hit_filtered += 1
                continue

            # Check weight threshold
            if data.get("weight", 0.0) < self.weight_threshold:
                weight_filtered += 1
                continue

            # Check prerequisites
            prerequisites_met = True
            for pred in self.entity_graph.predecessors(node_id):
                if self.entity_graph.nodes[pred].get("status", 0) != 2:
                    prerequisites_met = False
                    break

            if not prerequisites_met:
                prereq_filtered += 1
                continue

            available.append((node_id, data))

        self.logger.info(f"Found {len(available)} available nodes")
        self.logger.debug(f"Filtered nodes: {status_filtered} by status, {hit_filtered} by hit threshold, " +
                         f"{weight_filtered} by weight threshold, {prereq_filtered} by prerequisites, " +
                         f"{preset_filtered} preset nodes skipped")
        # log_messages.append(f"Found {len(available)} available nodes")
        # log_messages.append(f"Filtered nodes: {status_filtered} by status, {hit_filtered} by hit threshold, " +
        #                  f"{weight_filtered} by weight threshold, {prereq_filtered} by prerequisites")

        return available, log_messages

    def _select_node(self) -> tuple[str, dict[str, Any], list[str]] | None:
        """Select next node to query using scoring algorithm"""
        log_messages = []
        available_nodes, available_messages = self._get_available_nodes()
        log_messages.extend(available_messages)

        if not available_nodes:
            self.logger.info("No available nodes for selection")
            # log_messages.append("No available nodes for selection")
            return None, None, log_messages

        # Calculate PageRank
        pr = nx.pagerank(self.relation_graph)

        # Calculate scores for each node
        best_node_id = None
        best_data = None
        best_score = -float('inf')

        weights, entropies, topologies, communities = [], [], [], []

        for nid, data in available_nodes:
            weights.append(float(data.get("weight", 0.0)))
            entropies.append(float(data.get("uncertainty", 1.0)))
            topologies.append(float(pr.get(nid, 0)))

            if self.prev_node is None:
                communities.append(0.0)
            else:
                communities.append(self._calculate_community_score(nid, data))

        # Normalize scores
        normalize = lambda x: (np.array(x) - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) - np.min(x) != 0 else np.zeros_like(x)

        weights = normalize(weights)
        entropies = normalize(entropies)
        importance_score = normalize(weights * 5 + entropies)  # Weight more heavily
        topologies = normalize(topologies)
        communities = normalize(communities)

        # Calculate combined scores
        scores = self.alpha * importance_score + self.beta * topologies + self.gamma * communities

        # Select best node
        best_index = np.argmax(scores)
        best_node_id = available_nodes[best_index][0]
        best_data = available_nodes[best_index][1]
        best_score = scores[best_index]

        # Log scores
        for i, (nid, data) in enumerate(available_nodes):
            score_info = (
                f"Node {nid}: weight={weights[i]:.3f}, entropy={entropies[i]:.3f}, "
                f"topology={topologies[i]:.3f}, community={communities[i]:.3f}, score={scores[i]:.3f}"
            )
            self.logger.debug(score_info)
            # log_messages.append(score_info)

        # Update hit counter
        self.entity_graph.nodes[best_node_id]["hit"] += 1
        self.prev_node = best_node_id

        self.logger.info(f"Selected node {best_node_id} ({best_data.get('name', '')}) with score {best_score:.3f}")
        # log_messages.append(f"Selected node {best_node_id} ({best_data.get('name', '')}) with score {best_score:.3f}")
        return best_node_id, best_data, log_messages

    def _calculate_community_score(self, cand_id: str, cand_data: dict[str, Any]) -> float:
        """Calculate community coherence score"""
        if self.prev_node is None:
            return 1.0

        prev_comm = self.entity_graph.nodes[self.prev_node].get("community", None)
        if prev_comm is None:
            return 1.0

        # Get candidate neighbors
        candidate_neighbors = set(self.entity_graph.neighbors(cand_id)).union(
            set(self.relation_graph.predecessors(cand_id))
        )

        d_in = 0
        d_out = 0

        for neighbor in candidate_neighbors:
            neighbor_comm = self.entity_graph.nodes[neighbor].get("community", None)
            if neighbor_comm == prev_comm:
                d_in += 1
            else:
                d_out += 1

        epsilon = 1e-9
        eta = d_in / (d_in + d_out + epsilon)

        candidate_comm = cand_data.get("community")
        if candidate_comm is not None and candidate_comm == prev_comm:
            score = math.exp(eta)
        else:
            score = math.exp(-eta)

        return score

    def _process_user_message(
        self,
        hint_message: str,
        query_message: str,
        human_message: str,
        is_image_report: bool = False
    ) -> tuple[list[str], list[str], list[str]]:
        """
        Extract information from user message and update graph.

        Args:
            hint_message: Hint message for context
            query_message: Query message from AI
            human_message: User's response or image analysis report
            is_image_report: If True, treat human_message as image analysis report

        Returns:
            Tuple of (updated_node_ids, new_node_ids, log_messages)
        """
        messages = []
        log_messages = []
        extract_info = {"exist_nodes": [], "new_nodes": []}
        iteration = 0
        endpoint = False

        if is_image_report:
            self.logger.info("Processing image analysis report to extract information")
            # log_messages.append("Processing image analysis report to extract information")
        else:
            self.logger.info("Processing user message to extract information")
            # log_messages.append("Processing user message to extract information")

        while not endpoint and iteration < 10:
            if iteration == 0:
                if is_image_report:
                    # Use IMAGE_INFO_EXTRACTION for image reports
                    prompt = self.prompts.get(
                        "IMAGE_INFO_EXTRACTION",
                        purpose=self.target,
                        graph=self._serialize_nodes_with_value(self.entity_graph),
                        report=human_message,
                        language=self.language
                    )
                else:
                    # Use EXTRACT_INFO for regular user messages
                    prompt = self.prompts.get(
                        "EXTRACT_INFO",
                        purpose=self.target,
                        graph=self._serialize_nodes(self.entity_graph),
                        hint_message=hint_message,
                        query_message=query_message,
                        human_message=human_message,
                        language=self.language
                    )
            else:
                # Continue extraction for both types
                prompt = self.prompts.get("CONTINUE_EXTRACT_INFO")

            messages.append(HumanMessage(content=prompt))


            response = self.conv_model.invoke(messages, response_format=EXTRACT_INFO_SCHEMA)

            try:
                result = parse_json_response(response.content)
                exist_nodes = result.get("exist_nodes", [])
                new_nodes = result.get("new_nodes", [])

                extract_info["exist_nodes"].extend(exist_nodes)
                extract_info["new_nodes"].extend(new_nodes)

                self.logger.info(f"Iteration {iteration+1}: Extracted {len(exist_nodes)} existing nodes and {len(new_nodes)} new nodes")
                # log_messages.append(f"Iteration {iteration+1}: Extracted {len(exist_nodes)} existing nodes and {len(new_nodes)} new nodes")

                messages.append(response)
                endpoint = result.get("endpoint", True)
                if isinstance(endpoint, str):
                    endpoint = endpoint.lower() == "true"
                elif isinstance(endpoint, bool):
                    endpoint = endpoint
                else:
                    error_msg = f"Unexpected endpoint type: {type(endpoint)}"
                    self.logger.error(error_msg)
                    # log_messages.append(error_msg)
                    raise ValueError(error_msg)

                iteration += 1

                # Break if no new information
                if not exist_nodes and not new_nodes:
                    self.logger.info("No new information extracted, breaking extraction loop")
                    # log_messages.append("No new information extracted, breaking extraction loop")
                    break

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse extraction response: {e}"
                self.logger.error(error_msg)
                # log_messages.append(error_msg)
                break

        # Update existing nodes
        updated_nodes = []
        self.logger.info(f"Processing {len(extract_info.get('exist_nodes', []))} extracted nodes")
        for entry in extract_info.get("exist_nodes", []):
            node_id = entry.get("id")
            value = str(entry.get("value", "")).strip()
            # Convert confidential_level to float, handling string input from LLM
            try:
                confidential_level = float(entry.get("confidential_level", 0.0))
            except (ValueError, TypeError):
                confidential_level = 0.0

            self.logger.info(f"Checking extracted node: id={node_id}, value_preview={value[:30] if value else 'empty'}, in_graph={node_id in self.entity_graph.nodes if node_id else False}")

            if not value or node_id not in self.entity_graph.nodes:
                if not value:
                    self.logger.warning(f"Skipping node {node_id}: empty value")
                elif node_id not in self.entity_graph.nodes:
                    self.logger.warning(f"Skipping node {node_id}: not found in graph. Available nodes: {list(self.entity_graph.nodes)[:10]}...")
                continue

            updated_nodes.append(node_id)

            # Store value history (only for user messages, not image reports)
            if not is_image_report:
                if "value_history" not in self.entity_graph.nodes[node_id]:
                    self.entity_graph.nodes[node_id]["value_history"] = []
                self.entity_graph.nodes[node_id]["value_history"].append(
                    self.entity_graph.nodes[node_id].get("value", "")
                )

            # Update node value and confidential_level
            self.entity_graph.nodes[node_id]["value"] = value
            self.entity_graph.nodes[node_id]["confidential_level"] = confidential_level
            self.entity_graph.nodes[node_id]["last_updated_at"] = datetime.now()

            if "original_confidential_level" not in self.entity_graph.nodes[node_id]:
                self.entity_graph.nodes[node_id]["original_confidential_level"] = confidential_level
            else:
                # Convert stored value to float in case it was stored as string
                try:
                    stored_level = float(self.entity_graph.nodes[node_id]["original_confidential_level"])
                except (ValueError, TypeError):
                    stored_level = 0.0
                if confidential_level > stored_level:
                    self.entity_graph.nodes[node_id]["original_confidential_level"] = confidential_level

            # Update status based on confidential_level
            if is_image_report:
                # For image reports: direct mapping
                if confidential_level >= 0.7:
                    self.entity_graph.nodes[node_id]["status"] = 2  # High confidence
                elif confidential_level >= 0.4:
                    self.entity_graph.nodes[node_id]["status"] = 1  # Low confidence
            else:
                # For user messages: use threshold
                if confidential_level >= self.confidential_threshold:
                    self.entity_graph.nodes[node_id]["status"] = 2
                else:
                    self.entity_graph.nodes[node_id]["status"] = 1

            self.logger.info(f"Updated node {node_id} with value: {value[:50]}...")
            # log_messages.append(f"Updated node {node_id} with value: {value[:50]}...")

        # Add new nodes
        new_nodes = []
        for entry in extract_info.get("new_nodes", []):
            node_id = entry.get("id", f"v{uuid.uuid4().hex[:8]}")
            name = entry.get("name", "")
            value = str(entry.get("value", "")).strip()
            # Convert numeric fields to proper types, handling string input from LLM
            try:
                relevance = float(entry.get("relevance", 0.0))
            except (ValueError, TypeError):
                relevance = 0.0
            try:
                confidential_level = float(entry.get("confidential_level", 0.0))
            except (ValueError, TypeError):
                confidential_level = 0.0
            try:
                weight = float(entry.get("weight", 1.0))
            except (ValueError, TypeError):
                weight = 1.0
            try:
                uncertainty = float(entry.get("uncertainty", 1.0))
            except (ValueError, TypeError):
                uncertainty = 1.0

            if relevance < self.relevance_threshold:
                self.logger.info(f"Skipping node {name} with low relevance: {relevance:.2f}")
                # log_messages.append(f"Skipping node {name} with low relevance: {relevance:.2f}")
                continue

            if not name or not value:
                continue

            # Determine status based on confidential_level
            if is_image_report:
                status = 2 if confidential_level >= 0.7 else (1 if confidential_level >= 0.4 else 0)
            else:
                status = 2 if confidential_level >= self.confidential_threshold else 1

            now = datetime.now()
            new_node_data = {
                "id": node_id,
                "name": name,
                "description": entry.get("description", ""),
                "value": value,
                "weight": weight,
                "uncertainty": uncertainty,
                "confidential_level": confidential_level,
                "status": status,
                "hit": 1,
                "community": 0,  # Will be updated in clustering
                # 时间属性
                "extracted_at": now,
                "last_updated_at": now,
                "source": "conversation",
                "original_confidential_level": confidential_level,
            }

            self.entity_graph.add_node(node_id, **new_node_data)
            self.relation_graph.add_node(node_id, **new_node_data)
            new_nodes.append(node_id)

            self.logger.info(f"Added new node {node_id}: {name}")
            # log_messages.append(f"Added new node {node_id}: {name}")

        return updated_nodes, new_nodes, log_messages

    def _update_graph(self, updated_nodes: list[str], new_nodes: list[str]):
        """Update graph structure and weights based on new information"""
        log_messages = []

        # 1. Update weights and uncertainties of existing nodes (Entity Graph only)
        if updated_nodes:
            weight_update_messages = self._update_existing_node_weights(updated_nodes)
            log_messages.extend(weight_update_messages)

        # 2. Create new edges in Relation Graph for new nodes
        if new_nodes:
            self.logger.info(f"Creating relation edges for {len(new_nodes)} new nodes")
            relation_edge_messages = self._create_incremental_relation_edges(new_nodes, updated_nodes)
            log_messages.extend(relation_edge_messages)

        # 3. Re-cluster if new nodes were added
        if new_nodes:
            self.logger.info("Re-clustering graph due to new nodes")
            clustering_messages = self._clustering()
            log_messages.extend(clustering_messages)

        return log_messages

    def _update_existing_node_weights(self, updated_nodes: list[str]) -> list[str]:
        """Update weights and uncertainties of existing nodes based on new information"""
        log_messages = []

        # Get neighbors of updated nodes
        all_neighbors = []
        for node_id in updated_nodes:
            neighbors = list(self.entity_graph.neighbors(node_id))
            neighbors = [n for n in neighbors if self.entity_graph.nodes[n]["status"] in (0, 1)]
            all_neighbors.extend(neighbors)

        if not all_neighbors:
            self.logger.info("No neighbors to update")
            # log_messages.append("No neighbors to update")
            return log_messages

        # Update weights and uncertainties in chunks
        chunk_size = 20
        for i in range(0, len(all_neighbors), chunk_size):
            chunk = all_neighbors[i:i + chunk_size]

            relevant_nodes = "\n".join([
                f"node {n}, {self.entity_graph.nodes[n]['name']}, "
                f"initial weight {self.entity_graph.nodes[n]['weight']}, "
                f"initial uncertainty {self.entity_graph.nodes[n]['uncertainty']}"
                for n in chunk
            ])

            prompt = self.prompts.get(
                "UPDATE_GRAPH",
                purpose=self.target,
                collected=self._serialize_nodes_with_value(self.entity_graph),
                relevant_nodes=relevant_nodes
            )

            response = self.graph_model.invoke([SystemMessage(content=prompt)], response_format=UPDATE_GRAPH_SCHEMA)

            try:
                updates = parse_json_response(response.content)
                update_count = 0

                for update in updates:
                    node_id = update.get("id")
                    if node_id in self.entity_graph.nodes:
                        with contextlib.suppress(ValueError, TypeError):
                            self.entity_graph.nodes[node_id]["weight"] = float(update.get("weight", self.entity_graph.nodes[node_id]["weight"]))
                        with contextlib.suppress(ValueError, TypeError):
                            self.entity_graph.nodes[node_id]["uncertainty"] = float(update.get("uncertainty", self.entity_graph.nodes[node_id]["uncertainty"]))

                        update_count += 1
                        self.logger.info(f"Updated node {node_id}: {update.get('update_reason', 'No reason')}")
                        # log_messages.append(f"Updated node {node_id}: {update.get('update_reason', 'No reason')}")

                self.logger.info(f"Updated {update_count} nodes in chunk {i//chunk_size + 1}")
                # log_messages.append(f"Updated {update_count} nodes in chunk {i//chunk_size + 1}")

            except json.JSONDecodeError as e:
                error_msg = f"Failed to parse update response: {e}"
                self.logger.error(error_msg)

        return log_messages

    def _serialize_nodes(self, graph: nx.DiGraph) -> str:
        """Serialize graph nodes for prompts"""
        serialized_nodes = []
        serialize_keys = ["name", "description"]

        for node, attrs in graph.nodes(data=True):
            # Use node (the node ID) explicitly, then add other attributes
            attr_str = f"id: {node}, " + ", ".join(f"{key}: {attrs.get(key, '')}" for key in serialize_keys)
            serialized_nodes.append(attr_str)

        return "\n".join(serialized_nodes)

    def _serialize_nodes_with_value(self, graph: nx.DiGraph) -> str:
        """Serialize nodes with values for prompts"""
        serialized_nodes = []
        serialize_keys = ["name", "description", "value"]

        for _node, attrs in graph.nodes(data=True):
            if attrs.get("source") == "preset":
                continue
            if attrs.get("value"):
                attr_str = ", ".join(f"{key}: {attrs.get(key, '')}" for key in serialize_keys)
                if attrs.get("status", 0) == 1:
                    attr_str += ", this value is with low confidence"
                serialized_nodes.append(attr_str)

        return "\n".join(serialized_nodes)

    def _serialize_node_info(self, node: dict[str, Any]) -> str:
        """Serialize single node information"""
        info = f"- Name: {node.get('name', '')} (a short name of the entity)\n"
        info += f"- Description: {node.get('description', '')} (a detailed description of the node)\n"
        info += f"- Confidential level: {node.get('confidential_level', 0)} (confidence level [0, 1])\n"
        info += f"- Value: {node.get('value', '')} (the extracted value)\n"
        info += f"- Hit: {node.get('hit', 0)} (number of times queried)\n"
        info += f"- Status: {node.get('status', 0)} (0 for unknown, 1 for low confidence, 2 for high confidence)\n"
        return info

    def _format_key_nodes_for_prompt(self, key_nodes: list[dict]) -> str:
        """
        Format key nodes for injection into diagnosis prompt.

        Args:
            key_nodes: List of key node dictionaries from KeyNodeIdentifier

        Returns:
            Formatted string for prompt injection
        """
        if not key_nodes:
            return "No key diagnostic findings identified."

        lines = []
        lines.append("【Key Diagnostic Findings】")
        lines.append("")

        for i, node in enumerate(key_nodes, 1):
            name = node.get("name", "Unknown")
            value = node.get("value", "")
            combined_score = node.get("combined_score", 0)
            percentile = node.get("percentile_rank", 0)
            dims = node.get("dimension_scores", {})

            # Format: [1] Entity Name: Value (Combined Score: XX, Percentile: YY%)
            value_str = f": {value}" if value else ""
            lines.append(f"[{i}] {name}{value_str}")
            lines.append(f"    Combined Score: {combined_score:.2f}, Percentile: {percentile*100:.1f}%")

            # Add dimension breakdown
            lines.append("    Dimension Scores:")
            lines.append(f"      - Centrality: {dims.get('centrality', 0):.2f}")
            lines.append(f"      - Confidence: {dims.get('confidence', 0):.2f}")
            lines.append(f"      - Clinical Significance: {dims.get('clinical_significance', 0):.2f}")
            lines.append(f"      - Community Role: {dims.get('community_role', 0):.2f}")
            lines.append("")

        return "\n".join(lines)

    def _format_patient_text_records(self, patient_text_records: dict[str, str]) -> str:
        """格式化患者文本记录为LLM提示词上下文

        Args:
            patient_text_records: 患者文本记录字典

        Returns:
            格式化后的患者上下文字符串
        """
        if not patient_text_records:
            return ""

        lines = ["【患者背景信息】"]
        for field_name, text_value in patient_text_records.items():
            if text_value and text_value.strip():
                lines.append(f"\n{field_name}:\n{text_value}")

        return "\n".join(lines)

    def _select_candidate_nodes_for_relation_edges(
        self,
        new_nodes: list[str],
        updated_nodes: list[str]
    ) -> list[str]:
        """
        Select candidate nodes for creating relation edges with new nodes.
        
        Returns candidates including:
        1. updated_nodes
        2. 1-hop neighbors of updated_nodes in relation graph
        3. Existing nodes with semantic overlap (keyword matching)
        4. new_nodes themselves (for inter-new-node edges)
        
        Note: New nodes don't have community info yet, so community-based filtering is not used.
        
        Args:
            new_nodes: List of newly added node IDs
            updated_nodes: List of updated node IDs
            
        Returns:
            List of candidate node IDs (limited to 50)
        """
        candidates = set()

        # Strategy 1: Updated nodes as candidates (highest priority)
        candidates.update(updated_nodes)

        # Strategy 2: 1-hop neighbors of updated nodes in relation graph
        for node_id in updated_nodes:
            if node_id in self.relation_graph:
                neighbors = set(self.relation_graph.neighbors(node_id))
                predecessors = set(self.relation_graph.predecessors(node_id))
                successors = set(self.relation_graph.successors(node_id))
                candidates.update(neighbors | predecessors | successors)

        # Strategy 3: Existing nodes with semantic overlap (keyword matching)
        new_node_keywords = self._extract_keywords_from_nodes(new_nodes)
        for node_id, data in self.entity_graph.nodes(data=True):
            if node_id in new_nodes:  # Exclude new nodes themselves (will add separately)
                continue
            node_keywords = self._extract_keywords(data.get("name", ""), data.get("description", ""))
            if new_node_keywords & node_keywords:  # Has intersection
                candidates.add(node_id)

        # Strategy 4: Add new nodes themselves (for inter-new-node edges)
        candidates.update(new_nodes)

        # Limit candidates to 50 to avoid token limits
        if len(candidates) > 50:
            prioritized = []
            # Priority 1: updated_nodes
            prioritized.extend(updated_nodes)
            # Priority 2: 1-hop neighbors
            for node_id in updated_nodes:
                if node_id in self.relation_graph:
                    prioritized.extend(self.relation_graph.neighbors(node_id))
            # Priority 3: new nodes themselves
            prioritized.extend(new_nodes)
            # Priority 4: other candidates (keyword match)
            others = candidates - set(prioritized)
            prioritized.extend(others)

            # Deduplicate and truncate
            seen = set()
            unique_prioritized = []
            for node_id in prioritized:
                if node_id not in seen:
                    seen.add(node_id)
                    unique_prioritized.append(node_id)
            candidates = set(unique_prioritized[:50])

        return list(candidates)

    def _extract_keywords_from_nodes(self, node_ids: list[str]) -> set:
        """Extract keywords from a list of nodes"""
        keywords = set()
        for node_id in node_ids:
            if node_id in self.entity_graph:
                data = self.entity_graph.nodes[node_id]
                keywords.update(self._extract_keywords(data.get("name", ""), data.get("description", "")))
        return keywords

    def _extract_keywords(self, name: str, description: str) -> set:
        """
        Extract keywords from name and description.
        Uses NLTK stopwords for filtering common English words.
        """
        import re

        from nltk.corpus import stopwords
        stop_words = set(stopwords.words('english'))

        text = f"{name} {description}".lower()
        # Simple tokenization: split by spaces and punctuation
        words = re.findall(r'\b[a-z]+\b', text)
        # Filter stopwords and short words
        return set(w for w in words if w not in stop_words and len(w) > 2)

    def _filter_and_deduplicate_edges(self, edges: list[dict]) -> list[dict]:
        """
        Filter and deduplicate edge list.
        
        Rules:
        1. Remove self-loop edges (source == target)
        2. Remove duplicate (source, target) pairs (keep first)
        3. Filter out edges with missing nodes
        
        Args:
            edges: List of edge dictionaries
            
        Returns:
            Filtered and deduplicated edge list
        """
        seen = set()
        deduplicated = []
        skipped_self_loop = 0
        skipped_duplicate = 0
        skipped_missing_node = 0

        for edge in edges:
            source = edge.get("source")
            target = edge.get("target")

            # Check self-loop
            if source == target:
                skipped_self_loop += 1
                continue

            # Check if nodes exist
            if source not in self.entity_graph or target not in self.entity_graph:
                skipped_missing_node += 1
                continue

            # Check duplicate
            edge_key = (source, target)
            if edge_key in seen:
                skipped_duplicate += 1
                continue

            seen.add(edge_key)
            deduplicated.append(edge)

        self.logger.info(
            f"Edge filtering: {skipped_self_loop} self-loops, "
            f"{skipped_duplicate} duplicates, {skipped_missing_node} missing nodes"
        )

        return deduplicated

    def _create_incremental_relation_edges(self, new_nodes: list[str], updated_nodes: list[str]) -> list[str]:
        """
        Create medical relationship edges for new nodes in the relation graph.
        
        Args:
            new_nodes: List of newly added node IDs
            updated_nodes: List of updated node IDs
            
        Returns:
            List of log messages
        """
        log_messages = []

        # Step 1: Get all candidate nodes for edge creation
        candidate_ids = self._select_candidate_nodes_for_relation_edges(new_nodes, updated_nodes)

        all_entities = []
        for node_id in candidate_ids:
            node_data = self.entity_graph.nodes[node_id]
            all_entities.append({
                "id": node_id,
                "name": node_data["name"],
                "description": node_data["description"],
                "value": node_data.get("value", ""),
                "is_new": node_id in new_nodes  # Mark as new for LLM
            })

        if not all_entities:
            return log_messages

        self.logger.info(f"Selected {len(all_entities)} candidate nodes for relation edges")

        # Step 2: Use LLM to create all edges in one call
        edges = self._create_incremental_edges_with_llm(all_entities)

        # Step 3: Filter and deduplicate edges
        valid_edges = self._filter_and_deduplicate_edges(edges)

        # Step 4: Add edges to relation graph
        edge_count = 0
        for edge in valid_edges:
            source = edge.get("source")
            target = edge.get("target")

            # Double-check node existence
            if source not in self.relation_graph or target not in self.relation_graph:
                self.logger.warning(f"Skipping edge with missing node: {source} -> {target}")
                continue

            # Check if edge already exists in graph
            if self.relation_graph.has_edge(source, target):
                self.logger.debug(f"Edge already exists, skipping: {source} -> {target}")
                continue

            self.relation_graph.add_edge(source, target, **edge)
            edge_count += 1
            self.logger.info(f"Added relation edge: {source} -> {target}")

        self.logger.info(f"Total {edge_count} relation edges added (after deduplication)")
        return log_messages

    def _create_incremental_edges_with_llm(self, all_entities: list[dict]) -> list[dict]:
        """
        Use LLM to create incremental relation edges.
        
        Args:
            all_entities: List of all candidate entities with their attributes
            
        Returns:
            List of edge dictionaries from LLM
        """
        # Format all entities
        entities_str = ", ".join([
            f"id: {e['id']}, name: {e['name']}, value: {e.get('value', 'N/A')}, new: {e['is_new']}"
            for e in all_entities
        ])

        # Get graph summary
        existing_graph_summary = self._summarize_existing_graph_structure()

        # Get prompt
        prompt = self.prompts.get(
            "INCREMENTAL_RELATION_GRAPH_EDGES",
            purpose=self.target,
            all_entities=entities_str,
            existing_graph_summary=existing_graph_summary,
            language=self.language
        )

        messages = [HumanMessage(content=prompt)]
        response = self.graph_model.invoke(messages, response_format=ENTITY_EDGES_SCHEMA)

        try:
            result = parse_json_response(response.content)
            edges = result.get("edges", [])
            self.logger.info(f"LLM returned {len(edges)} incremental relation edges")
            return edges
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse incremental edges: {e}")
            return []

    def _summarize_existing_graph_structure(self) -> str:
        """
        Generate a summary of existing graph structure.
        
        Returns:
            String summary of graph statistics and important nodes
        """
        total_nodes = self.entity_graph.number_of_nodes()
        total_edges = self.entity_graph.number_of_edges()

        # Get high-weight nodes
        important_nodes = []
        for node_id, data in self.entity_graph.nodes(data=True):
            if float(data.get("weight", 0)) >= 0.8:
                important_nodes.append(f"{node_id}: {data['name']} (weight={data['weight']})")

        summary = f"Total nodes: {total_nodes}, Total edges: {total_edges}\n"
        summary += f"Important nodes: {', '.join(important_nodes[:20])}"
        return summary

