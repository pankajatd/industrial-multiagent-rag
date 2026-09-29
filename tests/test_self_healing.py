import numpy as np
import pytest
from src.graph import build_multiagent_graph
from src.tools.camera import SyntheticIndustrialGenerator

def test_self_healing_optical_degradation():
    """Test 1: Defocus blur and darkness should be flagged by VisionAgent and auto-recalibrated by SelfHealingAgent."""
    app = build_multiagent_graph()
    gen = SyntheticIndustrialGenerator(seed=123)
    frame, _ = gen.generate("crack")
    blurry = gen.inject_blur(frame, ksize=25)
    dark_blurry = gen.inject_underexposure(blurry, factor=0.15)

    state = {
        "frame_index": 10,
        "raw_frame": dark_blurry,
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    result = app.invoke(state)
    assert len(result.get("healing_actions", [])) >= 1
    assert any(h["failure_type"] == "image_degraded" for h in result["healing_actions"])
    assert result.get("status") in ["HEALED", "COMPLETED"]

def test_self_healing_feature_schema():
    """Test 2: Missing columns and NaNs in feature vector should be imputed with priors."""
    app = build_multiagent_graph()
    gen = SyntheticIndustrialGenerator(seed=456)
    frame, _ = gen.generate("normal")

    corrupted_features = {
        "contour_count": 1.0,
        "mean_intensity": np.nan,
        "glcm_contrast": np.nan
    }

    state = {
        "frame_index": 11,
        "raw_frame": frame,
        "features": corrupted_features,
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    result = app.invoke(state)
    assert any(h["failure_type"] == "feature_schema_invalid" for h in result.get("healing_actions", []))
    assert result.get("status") in ["HEALED", "COMPLETED"]

def test_self_healing_rag_low_relevance():
    """Test 3: Query drift yielding low relevance should trigger broad SOP fallback."""
    app = build_multiagent_graph()
    gen = SyntheticIndustrialGenerator(seed=789)
    frame, _ = gen.generate("scratch")

    state = {
        "frame_index": 12,
        "raw_frame": frame,
        "features": {col: 1.0 for col in [
            "contour_count", "total_area", "mean_area", "max_area", "mean_aspect_ratio",
            "mean_extent", "mean_solidity", "mean_eccentricity", "mean_intensity",
            "std_intensity", "intensity_range", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5",
            "hu_6", "hu_7", "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity"
        ]},
        "alert": {
            "frame_index": 12,
            "defect_type": "scratch",
            "confidence": 0.90,
            "severity_score": 6.5,
            "severity_level": "MEDIUM"
        },
        "search_queries": ["completely irrelevant obscure query nonexistent123"],
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    result = app.invoke(state)
    assert any(h["failure_type"] == "rag_low_relevance" for h in result.get("healing_actions", []))
    assert result.get("status") in ["HEALED", "COMPLETED"]

def test_self_healing_runtime_exception():
    """Test 4: Unhandled runtime exceptions should trigger safe fail-soft circuit breaker."""
    app = build_multiagent_graph()
    
    state = {
        "frame_index": 13,
        "raw_frame": None,
        "errors": [
            {
                "component": "diagnostic_agent",
                "error_type": "runtime_exception",
                "message": "ZeroDivisionError: simulated test crash",
                "resolved": False
            }
        ],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    result = app.invoke(state)
    assert any(h["failure_type"] == "runtime_exception" for h in result.get("healing_actions", []))
    assert result.get("status") in ["HEALED", "COMPLETED"]
