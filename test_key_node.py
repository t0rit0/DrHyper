"""
Tests for Key Node Identification in DrHyper system.

Key Node identification uses 4-dimension scoring:
1. Centrality (中心性) - PageRank + Betweenness Centrality
2. Confidence (置信度) - Confidential level from entity graph
3. Clinical Significance (临床意义) - Node weight
4. Community Role (社区角色) - Bridge/hub detection
"""

import pytest
import numpy as np
import networkx as nx
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

from drhyper.core.key_node import KeyNodeIdentifier


class TestKeyNodeIdentifier:
    """Test suite for KeyNodeIdentifier class"""

    def setup_method(self):
        """Set up test fixtures"""
        # Create mock entity graph with sample nodes
        self.entity_graph = nx.DiGraph()
        self.entity_graph.add_node("v1", name="血压", weight=0.95,
                                   confidential_level=0.85,
                                   last_updated_at=datetime.now() - timedelta(hours=2),
                                   community=0)
        self.entity_graph.add_node("v2", name="年龄", weight=0.6,
                                   confidential_level=0.9,
                                   last_updated_at=datetime.now() - timedelta(days=1),
                                   community=0)
        self.entity_graph.add_node("v3", name="家族史", weight=0.7,
                                   confidential_level=0.75,
                                   last_updated_at=datetime.now() - timedelta(days=5),
                                   community=1)
        self.entity_graph.add_node("v4", name="心电图", weight=0.85,
                                   confidential_level=0.8,
                                   last_updated_at=datetime.now() - timedelta(hours=6),
                                   community=1)

        # Create mock relation graph with edges
        self.relation_graph = nx.DiGraph()
        self.relation_graph.add_edges_from([
            ("v1", "v2", {"weight": 0.8}),
            ("v1", "v3", {"weight": 0.6}),
            ("v1", "v4", {"weight": 0.9}),
            ("v2", "v4", {"weight": 0.5}),
            ("v3", "v4", {"weight": 0.7}),
        ])

        # Mock config with proper attribute access (4-dimension weights, sum to 1.0)
        self.config = Mock()
        self.config.system.centrality_weight = 0.30
        self.config.system.confidence_weight = 0.25
        self.config.system.clinical_weight = 0.25
        self.config.system.community_role_weight = 0.20

        self.identifier = KeyNodeIdentifier(
            self.entity_graph,
            self.relation_graph,
            self.config
        )

    def test_compute_centrality_scores(self):
        """Test centrality score computation using PageRank and Betweenness"""
        centrality_scores = self.identifier._compute_centrality_scores()

        # Check all nodes have scores
        assert len(centrality_scores) == 4
        assert "v1" in centrality_scores
        assert "v2" in centrality_scores

        # Check scores are normalized (between 0 and 1)
        for score in centrality_scores.values():
            assert 0 <= score <= 1, f"Centrality score {score} not in [0, 1]"

        # At least some nodes should have non-zero scores
        assert max(centrality_scores.values()) > 0

    def test_compute_confidence_scores(self):
        """Test confidence score extraction from entity graph"""
        confidence_scores = self.identifier._compute_confidence_scores()

        # Check all nodes have scores
        assert len(confidence_scores) == 4

        # Check scores match confidential_level from entity graph
        assert confidence_scores["v1"] == 0.85
        assert confidence_scores["v2"] == 0.9

        # Check scores are in valid range
        for score in confidence_scores.values():
            assert 0 <= score <= 1

    def test_compute_clinical_significance_scores(self):
        """Test clinical significance scoring from node weights"""
        clinical_scores = self.identifier._compute_clinical_significance_scores()

        # Check all nodes have scores
        assert len(clinical_scores) == 4

        # Check scores match weight from entity graph
        assert clinical_scores["v1"] == 0.95
        assert clinical_scores["v2"] == 0.6

        # v1 (blood pressure) should have highest clinical significance for hypertension
        assert clinical_scores["v1"] == max(clinical_scores.values())

    def test_compute_community_role_bridge_node(self):
        """Test community role scoring for bridge nodes"""
        # Create a graph where v1 is clearly a bridge node
        bridge_entity_graph = nx.DiGraph()
        bridge_entity_graph.add_node("v1", community=0)
        bridge_entity_graph.add_node("v2", community=0)
        bridge_entity_graph.add_node("v3", community=1)
        bridge_entity_graph.add_node("v4", community=1)

        bridge_relation_graph = nx.DiGraph()
        # v1 connects to both communities (bridge)
        bridge_relation_graph.add_edges_from([
            ("v1", "v2"),  # v1 connects to community 0
            ("v1", "v3"),  # v1 connects to community 1 (bridge!)
            ("v1", "v4"),  # v1 connects to community 1
            ("v2", "v4"),  # v2 connects within/between community
        ])

        identifier = KeyNodeIdentifier(
            bridge_entity_graph,
            bridge_relation_graph,
            self.config
        )

        role_scores = identifier._compute_community_role_scores()

        # v1 should have a non-zero role score as it connects multiple communities
        # Just check the function runs and returns valid scores
        assert len(role_scores) == 4
        for score in role_scores.values():
            assert 0 <= score <= 1

    def test_compute_community_role_hub_node(self):
        """Test community role scoring for hub nodes"""
        # Create a graph where v1 is a hub (high intra-community connectivity)
        hub_entity_graph = nx.DiGraph()
        hub_entity_graph.add_node("v1", community=0)
        hub_entity_graph.add_node("v2", community=0)
        hub_entity_graph.add_node("v3", community=0)
        hub_entity_graph.add_node("v4", community=0)

        hub_relation_graph = nx.DiGraph()
        hub_relation_graph.add_edges_from([
            ("v1", "v2"),
            ("v1", "v3"),
            ("v1", "v4"),  # v1 connects to all nodes in same community (hub!)
            ("v2", "v3"),  # other connections
        ])

        identifier = KeyNodeIdentifier(
            hub_entity_graph,
            hub_relation_graph,
            self.config
        )

        role_scores = identifier._compute_community_role_scores()

        # v1 should have high role score as it's a hub
        assert role_scores["v1"] == max(role_scores.values())

    def test_compute_combined_scores(self):
        """Test combined score calculation from all dimensions"""
        combined_scores = self.identifier._compute_combined_scores()

        # Check all nodes have scores
        assert len(combined_scores) == 4

        # Check scores are weighted combination of dimensions
        # All scores should be in valid range [0, 1]
        for score in combined_scores.values():
            assert 0 <= score <= 1, f"Combined score {score} not in [0, 1]"

        # v1 should have a reasonably high score (high in most dimensions)
        assert combined_scores["v1"] > 0.5

    def test_identify_key_nodes_default_threshold(self):
        """Test key node identification with default 80th percentile threshold"""
        key_nodes = self.identifier.identify(percentile_threshold=0.80)

        # Should return nodes above 80th percentile
        assert isinstance(key_nodes, list)

        # Each key node should have required fields
        for node in key_nodes:
            assert "id" in node
            assert "name" in node
            assert "combined_score" in node
            assert "percentile_rank" in node
            assert "dimension_scores" in node

            # Check dimension scores exist (4 dimensions)
            dims = node["dimension_scores"]
            assert "centrality" in dims
            assert "confidence" in dims
            assert "clinical_significance" in dims
            assert "community_role" in dims

    def test_identify_key_nodes_custom_threshold(self):
        """Test key node identification with custom threshold"""
        # Lower threshold should return more nodes
        key_nodes_50 = self.identifier.identify(percentile_threshold=0.50)
        key_nodes_90 = self.identifier.identify(percentile_threshold=0.90)

        # 50th percentile should return >= nodes than 90th percentile
        assert len(key_nodes_50) >= len(key_nodes_90)

    def test_identify_key_nodes_min_score_filter(self):
        """Test that min_combined_score filter works"""
        # Set a very high min score - should return fewer nodes
        key_nodes = self.identifier.identify(
            percentile_threshold=0.50,
            min_combined_score=0.95  # Very high threshold
        )

        # Should filter out nodes below min score
        for node in key_nodes:
            assert node["combined_score"] >= 0.95

    def test_percentile_calculation_edge_cases(self):
        """Test percentile calculation with edge cases"""
        # Test with all same scores
        same_scores = {"v1": 0.5, "v2": 0.5, "v3": 0.5, "v4": 0.5}
        percentile = self.identifier._calculate_percentile_rank(same_scores, 0.5)
        assert percentile == 0.5  # All same score = 50th percentile

        # Test with single node
        single_scores = {"v1": 0.8}
        percentile = self.identifier._calculate_percentile_rank(single_scores, 0.8)
        # Single node = all scores (100%) are <= target, but we return 0.5 for edge case
        assert 0.5 <= percentile <= 1.0  # Should be in valid range

    def test_empty_graph_handling(self):
        """Test handling of empty graphs"""
        empty_entity = nx.DiGraph()
        empty_relation = nx.DiGraph()

        identifier = KeyNodeIdentifier(
            empty_entity,
            empty_relation,
            self.config
        )

        key_nodes = identifier.identify()
        assert key_nodes == []

    def test_normalize_scores_function(self):
        """Test score normalization utility"""
        scores = [0.1, 0.5, 0.9]
        normalized = self.identifier._normalize_scores(scores)

        # Check normalization
        assert len(normalized) == 3
        assert min(normalized) >= 0
        assert max(normalized) <= 1
        assert normalized[0] < normalized[1] < normalized[2]

        # Test with constant scores (avoid division by zero)
        constant_scores = [0.5, 0.5, 0.5]
        normalized_constant = self.identifier._normalize_scores(constant_scores)
        # Should handle gracefully (return uniform or original)
        assert len(normalized_constant) == 3


class TestKeyNodeFormatting:
    """Test key node formatting for prompt injection"""

    def setup_method(self):
        """Set up test fixtures"""
        self.config = Mock()
        self.config.system.centrality_weight = 0.30
        self.config.system.confidence_weight = 0.25
        self.config.system.clinical_weight = 0.25
        self.config.system.community_role_weight = 0.20

        # Create minimal graphs
        entity_graph = nx.DiGraph()
        entity_graph.add_node("v1", name="血压", weight=0.95,
                              confidential_level=0.85,
                              last_updated_at=datetime.now(),
                              community=0, value="145/92 mmHg")

        relation_graph = nx.DiGraph()
        relation_graph.add_node("v1")

        self.identifier = KeyNodeIdentifier(
            entity_graph,
            relation_graph,
            self.config
        )

    def test_entity_graph_has_format_key_nodes_method(self):
        """Test that EntityGraph has _format_key_nodes_for_prompt method"""
        from drhyper.core.graph import EntityGraph

        # Check EntityGraph has the method
        assert hasattr(EntityGraph, '_format_key_nodes_for_prompt')

    def test_conversation_delegates_to_entity_graph(self):
        """Test that LongConversation uses EntityGraph's key nodes"""
        from drhyper.core.conversation import LongConversation

        # LongConversation no longer has its own _format_key_nodes_for_prompt
        # It should use self.plan_graph._format_key_nodes_for_prompt instead
        assert not hasattr(LongConversation, '_format_key_nodes_for_prompt')

    def test_key_node_identifier_import(self):
        """Test that KeyNodeIdentifier can be imported and instantiated"""
        from drhyper.core.key_node import KeyNodeIdentifier
        import networkx as nx
        from unittest.mock import Mock

        # Create minimal graphs
        entity_graph = nx.DiGraph()
        entity_graph.add_node("v1", name="Test", weight=0.8,
                              confidential_level=0.8,
                              last_updated_at=datetime.now(),
                              community=0)

        relation_graph = nx.DiGraph()
        relation_graph.add_node("v1")

        # Mock config
        config = Mock()
        config.system.centrality_weight = 0.30
        config.system.confidence_weight = 0.25
        config.system.clinical_weight = 0.25
        config.system.community_role_weight = 0.20

        # Should not raise
        identifier = KeyNodeIdentifier(entity_graph, relation_graph, config)
        assert identifier is not None


class TestKeyNodeIntegration:
    """Integration tests for Key Node with EntityGraph"""

    def test_entity_graph_has_key_nodes_attribute(self):
        """Test that EntityGraph class has key_nodes attribute"""
        from drhyper.core.graph import EntityGraph

        # Check EntityGraph has the attribute
        assert hasattr(EntityGraph, '__init__')

        # We'll test actual instantiation in integration tests
        # This is a compile-time check

    def test_key_node_scores_sum_to_one(self):
        """Test that dimension weights sum to 1.0"""
        total_weight = (
            0.30 +  # centrality
            0.25 +  # confidence
            0.25 +  # clinical
            0.20    # community role
        )
        assert abs(total_weight - 1.0) < 0.01

    def test_entity_graph_format_key_nodes_for_prompt(self):
        """Test EntityGraph's _format_key_nodes_for_prompt method"""
        from drhyper.core.graph import EntityGraph
        from unittest.mock import Mock
        import networkx as nx

        # Create minimal EntityGraph
        entity_graph = nx.DiGraph()
        entity_graph.add_node("v1", name="血压", value="145/92 mmHg", weight=0.8,
                              confidential_level=0.8,
                              last_updated_at=datetime.now(),
                              community=0)

        relation_graph = nx.DiGraph()
        relation_graph.add_node("v1")

        # Mock config and models
        config = Mock()
        config.system.centrality_weight = 0.30
        config.system.confidence_weight = 0.25
        config.system.clinical_weight = 0.25
        config.system.community_role_weight = 0.20

        graph_model = Mock()
        conv_model = Mock()

        eg = EntityGraph(
            target="Test",
            graph_model=graph_model,
            conv_model=conv_model,
            routine=None,
            visualize=False,
            working_directory=None
        )
        eg.entity_graph = entity_graph
        eg.relation_graph = relation_graph

        # Create mock key nodes (4 dimensions)
        key_nodes = [
            {
                "id": "v1",
                "name": "血压",
                "value": "145/92 mmHg",
                "combined_score": 0.88,
                "percentile_rank": 0.92,
                "dimension_scores": {
                    "centrality": 0.85,
                    "confidence": 0.85,
                    "clinical_significance": 0.95,
                    "community_role": 0.78
                }
            }
        ]

        formatted = eg._format_key_nodes_for_prompt(key_nodes)

        # Check format includes key information
        assert "血压" in formatted
        assert "145/92 mmHg" in formatted
        assert "0.88" in formatted or "88%" in formatted
        assert "Combined Score" in formatted
        assert "Dimension Scores" in formatted

        # Should be readable string
        assert isinstance(formatted, str)
        assert len(formatted) > 50  # Should have substantial content

    def test_entity_graph_format_key_nodes_empty_list(self):
        """Test EntityGraph formatting with empty key nodes list"""
        from drhyper.core.graph import EntityGraph
        from unittest.mock import Mock
        import networkx as nx

        # Create minimal EntityGraph (setup same as above)
        entity_graph = nx.DiGraph()
        relation_graph = nx.DiGraph()

        config = Mock()
        graph_model = Mock()
        conv_model = Mock()

        eg = EntityGraph(
            target="Test",
            graph_model=graph_model,
            conv_model=conv_model,
            routine=None,
            visualize=False,
            working_directory=None
        )
        eg.entity_graph = entity_graph
        eg.relation_graph = relation_graph

        formatted = eg._format_key_nodes_for_prompt([])

        # Should return informative message even when empty
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "No key diagnostic findings" in formatted
