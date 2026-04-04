"""
Key Node Identification for DrHyper system.

Identifies diagnostically significant nodes using 4-dimension scoring:
1. Centrality (中心性) - PageRank + Betweenness Centrality from relation graph
2. Confidence (置信度) - Confidential level from entity graph
3. Clinical Significance (临床意义) - Node weight from entity graph
4. Community Role (社区角色) - Bridge/hub detection in relation graph
"""

from typing import Any

import networkx as nx
import numpy as np

from drhyper.config.settings import ConfigManager


class KeyNodeIdentifier:
    """
    Identifies key diagnostic nodes using 4-dimension scoring.

    Key nodes are those that are most diagnostically significant based on:
    - Network centrality in the relation graph
    - Confidence in the extracted information
    - Clinical significance for the diagnostic target
    - Community role (bridge or hub nodes)
    """

    # Default dimension weights (sum to 1.0)
    DEFAULT_CENTRALITY_WEIGHT = 0.30
    DEFAULT_CONFIDENCE_WEIGHT = 0.25
    DEFAULT_CLINICAL_WEIGHT = 0.25
    DEFAULT_COMMUNITY_ROLE_WEIGHT = 0.20

    def __init__(self, entity_graph: nx.DiGraph, relation_graph: nx.DiGraph, config: ConfigManager):
        """
        Initialize Key Node Identifier.
        
        Args:
            entity_graph: Entity graph with node attributes (weight, confidential_level, etc.)
            relation_graph: Relation graph for centrality and community analysis
            config: Configuration manager with dimension weights
        """
        self.entity_graph = entity_graph
        self.relation_graph = relation_graph
        self.config = config

        # Get dimension weights from config (with defaults)
        self.centrality_weight = getattr(config.system, 'centrality_weight', self.DEFAULT_CENTRALITY_WEIGHT)
        self.confidence_weight = getattr(config.system, 'confidence_weight', self.DEFAULT_CONFIDENCE_WEIGHT)
        self.clinical_weight = getattr(config.system, 'clinical_weight', self.DEFAULT_CLINICAL_WEIGHT)
        self.community_role_weight = getattr(config.system, 'community_role_weight', self.DEFAULT_COMMUNITY_ROLE_WEIGHT)

        # Normalize weights if they don't sum to 1.0
        total = (self.centrality_weight + self.confidence_weight +
                 self.clinical_weight + self.community_role_weight)
        if total > 0 and abs(total - 1.0) > 0.01:
            self.centrality_weight /= total
            self.confidence_weight /= total
            self.clinical_weight /= total
            self.community_role_weight /= total

    def identify(
        self,
        percentile_threshold: float = 0.80,
        min_combined_score: float = 0.60
    ) -> list[dict[str, Any]]:
        """
        Identify key nodes based on 4-dimension scoring.

        Args:
            percentile_threshold: Minimum percentile rank (0-1) for key node identification
            min_combined_score: Minimum absolute combined score (0-1) for key node identification

        Returns:
            List of key node dictionaries with scores and metadata
        """
        # Handle empty graphs
        if self.entity_graph.number_of_nodes() == 0:
            return []

        # Compute all dimension scores
        centrality_scores = self._compute_centrality_scores()
        confidence_scores = self._compute_confidence_scores()
        clinical_scores = self._compute_clinical_significance_scores()
        community_role_scores = self._compute_community_role_scores()

        # Get all node IDs from entity graph
        node_ids = list(self.entity_graph.nodes())

        # Compute combined scores for each node
        combined_scores = {}
        dimension_scores_map = {}

        for node_id in node_ids:
            # Get individual dimension scores (default to 0 if not found)
            centrality = centrality_scores.get(node_id, 0.0)
            confidence = confidence_scores.get(node_id, 0.0)
            clinical = clinical_scores.get(node_id, 0.0)
            community_role = community_role_scores.get(node_id, 0.0)

            # Store dimension scores
            dimension_scores_map[node_id] = {
                "centrality": centrality,
                "confidence": confidence,
                "clinical_significance": clinical,
                "community_role": community_role
            }

            # Compute weighted combined score
            combined = (
                self.centrality_weight * centrality +
                self.confidence_weight * confidence +
                self.clinical_weight * clinical +
                self.community_role_weight * community_role
            )
            combined_scores[node_id] = combined

        # Calculate percentile ranks
        scores_array = np.array(list(combined_scores.values()))

        # Identify key nodes (above both percentile AND absolute thresholds)
        key_nodes = []
        for node_id in node_ids:
            combined_score = combined_scores[node_id]
            percentile_rank = self._calculate_percentile_rank(combined_scores, combined_score)

            # Apply both thresholds
            if percentile_rank >= percentile_threshold and combined_score >= min_combined_score:
                node_data = self.entity_graph.nodes[node_id]

                key_nodes.append({
                    "id": node_id,
                    "name": node_data.get("name", "Unknown"),
                    "value": node_data.get("value", ""),
                    "description": node_data.get("description", ""),
                    "combined_score": round(combined_score, 4),
                    "percentile_rank": round(percentile_rank, 4),
                    "dimension_scores": {
                        k: round(v, 4) for k, v in dimension_scores_map[node_id].items()
                    }
                })

        # Sort by combined score (descending)
        key_nodes.sort(key=lambda x: x["combined_score"], reverse=True)

        return key_nodes

    def _compute_centrality_scores(self) -> dict[str, float]:
        """
        Compute centrality scores using PageRank + Betweenness Centrality.
        
        Returns:
            Dictionary mapping node_id -> normalized centrality score
        """
        if self.relation_graph.number_of_nodes() == 0:
            return {}

        # Compute PageRank
        try:
            pagerank = nx.pagerank(self.relation_graph, max_iter=100)
        except nx.PowerIterationFailedConvergence:
            # Fallback for graphs that don't converge
            pagerank = {node: 1.0 / self.relation_graph.number_of_nodes()
                       for node in self.relation_graph.nodes()}

        # Compute Betweenness Centrality
        betweenness = nx.betweenness_centrality(self.relation_graph)

        # Get all node IDs from entity graph (may have nodes not in relation graph)
        all_node_ids = list(self.entity_graph.nodes())

        # Combine scores (average of PageRank and Betweenness)
        combined = {}
        for node_id in all_node_ids:
            pr = pagerank.get(node_id, 0.0)
            bc = betweenness.get(node_id, 0.0)
            combined[node_id] = (pr + bc) / 2.0

        # Normalize scores
        return self._normalize_scores_dict(combined)

    def _compute_confidence_scores(self) -> dict[str, float]:
        """
        Extract confidence scores from entity graph confidential_level attribute.

        Returns:
            Dictionary mapping node_id -> confidence score
        """
        confidence_scores = {}

        for node_id in self.entity_graph.nodes():
            node_data = self.entity_graph.nodes[node_id]
            confidence = node_data.get("confidential_level", 0.5)
            confidence_scores[node_id] = float(confidence)

        return confidence_scores

    def _compute_clinical_significance_scores(self) -> dict[str, float]:
        """
        Extract clinical significance scores from entity graph weight attribute.
        
        Returns:
            Dictionary mapping node_id -> clinical significance score
        """
        clinical_scores = {}

        for node_id in self.entity_graph.nodes():
            node_data = self.entity_graph.nodes[node_id]
            weight = node_data.get("weight", 0.5)
            clinical_scores[node_id] = float(weight)

        return clinical_scores

    def _compute_community_role_scores(self) -> dict[str, float]:
        """
        Compute community role scores for bridge and hub detection.
        
        Bridge nodes: Connect multiple communities (high inter-community edges)
        Hub nodes: High intra-community connectivity
        
        Returns:
            Dictionary mapping node_id -> community role score
        """
        if self.relation_graph.number_of_nodes() == 0:
            return {}

        role_scores = {}

        # Get community assignments from entity graph
        communities = {}
        for node_id in self.entity_graph.nodes():
            node_data = self.entity_graph.nodes[node_id]
            communities[node_id] = node_data.get("community", 0)

        # For each node, compute bridge and hub scores
        for node_id in self.entity_graph.nodes():
            node_community = communities.get(node_id, 0)

            # Get neighbors in relation graph
            neighbors = list(self.relation_graph.neighbors(node_id))
            if not neighbors:
                neighbors = list(self.relation_graph.predecessors(node_id))

            if not neighbors:
                # Isolated node - low role score
                role_scores[node_id] = 0.3
                continue

            # Count inter-community and intra-community connections
            inter_community_count = 0
            intra_community_count = 0

            for neighbor in neighbors:
                neighbor_community = communities.get(neighbor, 0)
                if neighbor_community == node_community:
                    intra_community_count += 1
                else:
                    inter_community_count += 1

            total_connections = inter_community_count + intra_community_count

            if total_connections == 0:
                role_scores[node_id] = 0.3
                continue

            # Bridge score: proportion of inter-community connections
            bridge_score = inter_community_count / total_connections

            # Hub score: proportion of intra-community connections, weighted by degree
            hub_score = (intra_community_count / total_connections) * min(1.0, total_connections / 3.0)

            # Combined role score: favor bridges slightly, but also reward hubs
            # Bridges connect different clinical concepts (important for diagnosis)
            # Hubs are central within a clinical domain
            role_scores[node_id] = max(
                0.6 * bridge_score + 0.4 * hub_score,  # Bridge-focused
                0.4 * bridge_score + 0.6 * hub_score   # Hub-focused
            )

        # Normalize scores
        return self._normalize_scores_dict(role_scores)

    def _compute_combined_scores(self) -> dict[str, float]:
        """
        Compute combined scores from all dimensions.

        Returns:
            Dictionary mapping node_id -> combined score
        """
        centrality = self._compute_centrality_scores()
        confidence = self._compute_confidence_scores()
        clinical = self._compute_clinical_significance_scores()
        community_role = self._compute_community_role_scores()

        combined = {}
        for node_id in self.entity_graph.nodes():
            score = (
                self.centrality_weight * centrality.get(node_id, 0.0) +
                self.confidence_weight * confidence.get(node_id, 0.0) +
                self.clinical_weight * clinical.get(node_id, 0.0) +
                self.community_role_weight * community_role.get(node_id, 0.0)
            )
            combined[node_id] = score

        return combined

    def _calculate_percentile_rank(self, scores: dict[str, float], target_score: float) -> float:
        """
        Calculate percentile rank for a given score.
        
        Percentile rank = proportion of scores <= target_score
        
        Args:
            scores: Dictionary of node_id -> score
            target_score: Score to calculate percentile for
            
        Returns:
            Percentile rank (0-1)
        """
        if not scores:
            return 0.0

        scores_array = np.array(list(scores.values()))

        # Handle edge case: all scores are the same
        if np.all(scores_array == target_score):
            return 0.5  # Middle percentile

        # Calculate proportion of scores <= target
        count_below = np.sum(scores_array <= target_score)
        percentile = count_below / len(scores_array)

        return float(percentile)

    def _normalize_scores(self, scores: list[float]) -> np.ndarray:
        """
        Normalize scores to [0, 1] range using min-max normalization.
        
        Args:
            scores: List of raw scores
            
        Returns:
            Array of normalized scores
        """
        scores_array = np.array(scores, dtype=float)

        min_score = np.min(scores_array)
        max_score = np.max(scores_array)

        # Handle constant scores (avoid division by zero)
        if max_score - min_score < 1e-10:
            # Return uniform scores
            return np.ones_like(scores_array) * 0.5

        normalized = (scores_array - min_score) / (max_score - min_score)
        return normalized

    def _normalize_scores_dict(self, scores: dict[str, float]) -> dict[str, float]:
        """
        Normalize a dictionary of scores to [0, 1] range.
        
        Args:
            scores: Dictionary mapping node_id -> score
            
        Returns:
            Dictionary mapping node_id -> normalized score
        """
        if not scores:
            return {}

        values = list(scores.values())
        normalized_values = self._normalize_scores(values)

        return {key: float(norm) for key, norm in zip(scores.keys(), normalized_values)}
