import pytest
from src.graph import build_multiagent_graph
from src.tools.camera import SyntheticIndustrialGenerator

def test_multiagent_graph_normal_frame():
    app = build_multiagent_graph()
    gen = SyntheticIndustrialGenerator(seed=10)
    frame, _ = gen.generate("normal")

    state = {
        "frame_index": 1,
        "raw_frame": frame,
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    result = app.invoke(state)
    assert result.get("status") in ["COMPLETED", "HEALTHY"]
    assert "alert" in result
    assert result["alert"]["defect_type"] == "normal"
    assert result["alert"]["severity_level"] == "PASS"
    assert result.get("work_order") is None  # PASS frame does not trigger RAG work order

def test_multiagent_graph_critical_crack():
    app = build_multiagent_graph()
    gen = SyntheticIndustrialGenerator(seed=42)
    frame, _ = gen.generate("crack")

    state = {
        "frame_index": 2,
        "raw_frame": frame,
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    result = app.invoke(state)
    assert result.get("status") in ["COMPLETED", "HEALTHY"]
    assert "alert" in result
    assert result["alert"]["defect_type"] == "crack"
    assert result["alert"]["severity_score"] >= 4.0
    assert result.get("work_order") is not None
    assert "WO-M" in result["work_order"]["work_order_id"]
    assert result.get("human_approved") is True
