"""
Integration tests for Key Nodes end-to-end flow.

Tests that key nodes:
1. Are correctly identified by EntityGraph
2. Are injected into the accomplishment hint prompt
3. Are included in diagnostic report generation
4. Are persisted and restored from cache
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock
import networkx as nx


class TestKeyNodesEndToEnd:
    """End-to-end tests for key nodes system"""

    def test_key_nodes_identified_when_accomplish(self):
        """Test that key nodes are identified when all nodes are processed"""
        from drhyper.core.graph import EntityGraph
        from drhyper.config.settings import ConfigManager

        # Create EntityGraph with mock models
        graph_model = Mock()
        graph_model.invoke = Mock(return_value=Mock(content="Test hint"))
        conv_model = Mock()

        eg = EntityGraph(
            target="Hypertension diagnosis",
            graph_model=graph_model,
            conv_model=conv_model,
            routine=None,
            visualize=False,
            working_directory=None
        )

        # Create entity graph with multiple nodes
        eg.entity_graph = nx.DiGraph()
        eg.entity_graph.add_node("v1", name="血压", value="145/92 mmHg", weight=0.9,
                                  temporal_confidence=0.85, freshness=0.9,
                                  last_updated_at=datetime.now(),
                                  community=0, status=2, hit=3)
        eg.entity_graph.add_node("v2", name="年龄", value="55 岁", weight=0.7,
                                  temporal_confidence=0.9, freshness=0.95,
                                  last_updated_at=datetime.now(),
                                  community=0, status=2, hit=2)
        eg.entity_graph.add_node("v3", name="症状", value="头痛，头晕", weight=0.8,
                                  temporal_confidence=0.8, freshness=0.85,
                                  last_updated_at=datetime.now(),
                                  community=1, status=2, hit=2)

        # Add edges to create relationships
        eg.entity_graph.add_edge("v1", "v2", weight=0.7)
        eg.entity_graph.add_edge("v1", "v3", weight=0.8)

        eg.relation_graph = nx.DiGraph()
        for node in eg.entity_graph.nodes():
            eg.relation_graph.add_node(node)

        # Set all nodes to high confidence (status=2) so accomplishment triggers
        for node_id in eg.entity_graph.nodes():
            eg.entity_graph.nodes[node_id]["status"] = 2

        # Call get_hint_message which should trigger key node identification
        hint_message, accomplish, logs = eg.get_hint_message()

        # Verify accomplishment is True
        assert accomplish is True

        # Verify key nodes were identified
        assert hasattr(eg, 'key_nodes')
        assert len(eg.key_nodes) > 0

        # Verify key nodes have required fields
        for node in eg.key_nodes:
            assert 'name' in node
            assert 'combined_score' in node
            assert 'dimension_scores' in node
            assert 'centrality' in node['dimension_scores']
            assert 'confidence' in node['dimension_scores']

    def test_key_nodes_injected_into_hint_prompt(self):
        """Test that key nodes are included in the accomplishment hint prompt"""
        from drhyper.core.graph import EntityGraph
        from drhyper.prompts.templates import ConversationPrompts

        # Create EntityGraph with mock models
        graph_model = Mock()
        conv_model = Mock()

        eg = EntityGraph(
            target="Hypertension diagnosis",
            graph_model=graph_model,
            conv_model=conv_model,
            routine=None,
            visualize=False,
            working_directory=None
        )

        # Create entity graph
        eg.entity_graph = nx.DiGraph()
        eg.entity_graph.add_node("v1", name="血压", value="145/92 mmHg", weight=0.9,
                                  temporal_confidence=0.85, freshness=0.9,
                                  last_updated_at=datetime.now(),
                                  community=0, status=2, hit=3)
        eg.entity_graph.add_node("v2", name="年龄", value="55 岁", weight=0.7,
                                  temporal_confidence=0.9, freshness=0.95,
                                  last_updated_at=datetime.now(),
                                  community=0, status=2, hit=2)

        eg.relation_graph = nx.DiGraph()
        for node in eg.entity_graph.nodes():
            eg.relation_graph.add_node(node)

        # Mock the graph_model.invoke to capture the prompt
        captured_prompt = None

        def mock_invoke(messages):
            nonlocal captured_prompt
            captured_prompt = messages[0].content if hasattr(messages[0], 'content') else str(messages[0])
            return Mock(content="Test accomplishment hint")

        graph_model.invoke = mock_invoke

        # Set all nodes to high confidence
        for node_id in eg.entity_graph.nodes():
            eg.entity_graph.nodes[node_id]["status"] = 2

        # Call get_hint_message
        hint_message, accomplish, logs = eg.get_hint_message()

        # Verify the prompt includes key diagnostic findings section
        assert captured_prompt is not None
        assert "Key Diagnostic Findings" in captured_prompt or "关键诊断发现" in captured_prompt

        # Verify key nodes info is in the prompt
        assert "血压" in captured_prompt or "145/92" in captured_prompt

    def test_key_nodes_format_for_prompt(self):
        """Test the key nodes formatting function"""
        from drhyper.core.graph import EntityGraph
        from unittest.mock import Mock
        import networkx as nx

        # Setup EntityGraph
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
        eg.entity_graph = nx.DiGraph()
        eg.relation_graph = nx.DiGraph()

        # Create mock key nodes with all dimension scores
        key_nodes = [
            {
                "id": "v1",
                "name": "Hypertension",
                "value": "Stage 2 (145/92 mmHg)",
                "combined_score": 0.88,
                "percentile_rank": 0.92,
                "dimension_scores": {
                    "centrality": 0.85,
                    "confidence": 0.85,
                    "temporal_correlation": 0.90,
                    "clinical_significance": 0.95,
                    "community_role": 0.78
                }
            },
            {
                "id": "v2",
                "name": "Age",
                "value": "55 years",
                "combined_score": 0.72,
                "percentile_rank": 0.85,
                "dimension_scores": {
                    "centrality": 0.70,
                    "confidence": 0.90,
                    "temporal_correlation": 0.65,
                    "clinical_significance": 0.75,
                    "community_role": 0.60
                }
            }
        ]

        formatted = eg._format_key_nodes_for_prompt(key_nodes)

        # Verify format
        assert "Key Diagnostic Findings" in formatted
        assert "Hypertension" in formatted
        assert "Stage 2" in formatted
        assert "Combined Score: 0.88" in formatted
        assert "Percentile: 92.0%" in formatted
        assert "Dimension Scores:" in formatted
        assert "Centrality: 0.85" in formatted
        assert "Confidence: 0.85" in formatted
        assert "Temporal Correlation: 0.90" in formatted
        assert "Clinical Significance: 0.95" in formatted
        assert "Community Role: 0.78" in formatted

    def test_key_nodes_persistence_in_cache_dict(self):
        """Test that key nodes are persisted in to_cache_dict"""
        from drhyper.core.conversation import LongConversation
        from drhyper.core.graph import EntityGraph
        from unittest.mock import Mock
        import networkx as nx

        # Create LongConversation with mock models
        graph_model = Mock()
        conv_model = Mock()

        conv = LongConversation(
            target="Hypertension diagnosis",
            conv_model=conv_model,
            graph_model=graph_model,
            routine=None,
            visualize=False,
            working_directory=None
        )

        # Manually set key nodes
        conv.plan_graph.key_nodes = [
            {
                "id": "v1",
                "name": "血压",
                "value": "145/92 mmHg",
                "combined_score": 0.88,
                "percentile_rank": 0.92,
                "dimension_scores": {
                    "centrality": 0.85,
                    "confidence": 0.85,
                    "temporal_correlation": 0.90,
                    "clinical_significance": 0.95,
                    "community_role": 0.78
                }
            }
        ]
        conv.plan_graph.step = 5
        conv.plan_graph.accomplish = True

        # Convert to cache dict
        cache_dict = conv.to_cache_dict()

        # Verify key nodes are in cache
        assert "graph_state" in cache_dict
        assert "key_nodes" in cache_dict["graph_state"]
        assert len(cache_dict["graph_state"]["key_nodes"]) == 1
        assert cache_dict["graph_state"]["key_nodes"][0]["name"] == "血压"

        # Verify version is updated
        assert cache_dict["metadata"]["version"] == "2.3"

    def test_key_nodes_restored_from_cache_dict(self):
        """Test that key nodes are restored from cache_dict"""
        from drhyper.core.conversation import LongConversation
        from unittest.mock import Mock
        import networkx as nx

        # Create a cache dict with key nodes
        cache_dict = {
            "messages": [],
            "entire_messages": [],
            "current_hint": "",
            "step": 5,
            "think_history": [],
            "message_reserve_turns": 2,
            "target": "Hypertension diagnosis",
            "routine": None,
            "visualize": False,
            "working_directory": None,
            "stream": False,
            "entity_graph": nx.node_link_data(nx.DiGraph()),
            "relation_graph": nx.node_link_data(nx.DiGraph()),
            "graph_state": {
                "step": 5,
                "accomplish": True,
                "prev_node": None,
                "key_nodes": [
                    {
                        "id": "v1",
                        "name": "血压",
                        "value": "145/92 mmHg",
                        "combined_score": 0.88,
                        "percentile_rank": 0.92,
                        "dimension_scores": {
                            "centrality": 0.85,
                            "confidence": 0.85,
                            "temporal_correlation": 0.90,
                            "clinical_significance": 0.95,
                            "community_role": 0.78
                        }
                    }
                ]
            },
            "metadata": {
                "cached_at": datetime.now().isoformat(),
                "version": "2.3",
                "message_count": 0,
                "entity_graph_nodes": 0,
                "entity_graph_edges": 0,
                "relation_graph_nodes": 0,
                "relation_graph_edges": 0
            }
        }

        # Mock models
        conv_model = Mock()
        graph_model = Mock()

        # Restore from cache
        restored_conv = LongConversation.from_cache_dict(
            cache_dict,
            conv_model=conv_model,
            graph_model=graph_model
        )

        # Verify key nodes are restored
        assert hasattr(restored_conv.plan_graph, 'key_nodes')
        assert len(restored_conv.plan_graph.key_nodes) == 1
        assert restored_conv.plan_graph.key_nodes[0]["name"] == "血压"
        assert restored_conv.plan_graph.key_nodes[0]["combined_score"] == 0.88

    def test_key_nodes_empty_list_handled(self):
        """Test that empty key nodes list is handled gracefully"""
        from drhyper.core.graph import EntityGraph
        from unittest.mock import Mock
        import networkx as nx

        # Setup EntityGraph
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
        eg.entity_graph = nx.DiGraph()
        eg.relation_graph = nx.DiGraph()

        # Format empty key nodes
        formatted = eg._format_key_nodes_for_prompt([])

        assert formatted == "No key diagnostic findings identified."


class TestKeyNodesInReportGeneration:
    """Tests for key nodes usage in report generation"""

    def test_report_template_has_key_nodes_placeholder(self):
        """Test that report template includes key_nodes_info placeholder"""
        from pathlib import Path

        template_path = Path(__file__).parent.parent / "backend" / "prompts" / "report_template.txt"

        if template_path.exists():
            template_content = template_path.read_text()

            # Verify template has key_nodes_info placeholder
            assert "{key_nodes_info}" in template_content

            # Verify template instructs to use key findings
            assert "Key Diagnostic Findings" in template_content or "关键诊断发现" in template_content
        else:
            pytest.skip("Report template not found")

    def test_entity_graph_has_format_method_for_report(self):
        """Test that EntityGraph has the format method needed for report generation"""
        from drhyper.core.graph import EntityGraph

        assert hasattr(EntityGraph, '_format_key_nodes_for_prompt')

        # Verify it's callable
        import inspect
        assert inspect.ismethod(EntityGraph._format_key_nodes_for_prompt) or \
               inspect.isfunction(EntityGraph._format_key_nodes_for_prompt)
