import pytest
import numpy as np
from src.agents import VisionAgent, DiagnosticAgent, MaintenanceRAGAgent, QualityGateAgent
from src.tools.camera import SyntheticIndustrialGenerator
from src.tools.vector_store import VectorStore

def test_vision_agent():
    agent = VisionAgent()
    gen = SyntheticIndustrialGenerator(seed=1)
    frame, _ = gen.generate("normal")

    state = {
        "frame_index": 1,
        "raw_frame": frame,
        "errors": []
    }
    out = agent(state)
    assert "processed_frame" in out
    assert "mask" in out
    assert "features" in out
    assert len(out["features"]) == 21

def test_diagnostic_agent():
    agent = DiagnosticAgent()
    gen = SyntheticIndustrialGenerator(seed=2)
    frame, _ = gen.generate("crack")
    
    # Run vision first to get features
    vision = VisionAgent()
    v_out = vision({"frame_index": 2, "raw_frame": frame, "errors": []})
    
    diag_out = agent(v_out)
    assert "alert" in diag_out
    assert "defect_type" in diag_out["alert"]
    assert "severity_score" in diag_out["alert"]

def test_maintenance_rag_agent():
    vs = VectorStore()
    rag = MaintenanceRAGAgent(vs)
    state = {
        "frame_index": 3,
        "alert": {
            "defect_type": "crack",
            "severity_score": 8.0,
            "severity_level": "CRITICAL"
        },
        "errors": []
    }
    out = rag(state)
    assert "work_order" in out
    assert "safety_directives" in out["work_order"]
    assert len(out["work_order"]["repair_procedure"]) > 0

def test_quality_gate_agent():
    qg = QualityGateAgent()
    critical_state = {
        "alert": {"severity_level": "CRITICAL"},
        "work_order": {"work_order_id": "WO-TEST"}
    }
    out = qg(critical_state)
    assert out.get("requires_human_signoff") is True
    assert out.get("human_approved") is True
