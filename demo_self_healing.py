import numpy as np
from src.graph import build_multiagent_graph
from src.tools.camera import SyntheticIndustrialGenerator

def run_all_healing_scenarios():
    print("=" * 85)
    print(" INDUSTRIAL MULTI-AGENT PLATFORM: AUTONOMOUS SELF-HEALING DEMONSTRATION")
    print("=" * 85)
    print("Demonstrating 4 distinct classes of autonomous failure recovery handled by SelfHealingAgent:\n")

    app = build_multiagent_graph()
    generator = SyntheticIndustrialGenerator(seed=777)

    # ---------------------------------------------------------------------------------
    # SCENARIO 1: Sensor & Image Flaw Recovery (Defocus Blur & Low-Contrast Injection)
    # ---------------------------------------------------------------------------------
    print("\n" + "#" * 85)
    print(" [SCENARIO 1/4] OPTICAL SENSOR & IMAGE FLAW RECOVERY")
    print(" Failure Injected: Heavy Gaussian defocus blur + underexposure on camera feed.")
    print("#" * 85)
    
    clean_frame, _ = generator.generate("crack")
    blurry_frame = generator.inject_blur(clean_frame, ksize=25)
    degraded_frame = generator.inject_underexposure(blurry_frame, factor=0.18)

    state1 = {
        "frame_index": 201,
        "raw_frame": degraded_frame,
        "execution_log": ["[Test Harness] Injected degraded optical frame (severe blur + darkness)."],
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    out1 = app.invoke(state1)
    print("\nExecution Trace:")
    for l in out1.get("execution_log", []):
        print(f"  > {l}")
    print("\nSelf-Healing Actions Taken:")
    for h in out1.get("healing_actions", []):
        print(f"  [HEALED] Component: {h['target_component']} | Action: {h['action_taken']}")
    print(f"Final Defect Diagnosis: {out1.get('alert', {}).get('defect_type', '').upper()} (Severity: {out1.get('alert', {}).get('severity_level')})")
    assert out1.get("status") in ["HEALED", "COMPLETED"], "Scenario 1 failed to heal!"

    # ---------------------------------------------------------------------------------
    # SCENARIO 2: Feature & ML Schema Healing (Missing Columns & NaN Values)
    # ---------------------------------------------------------------------------------
    print("\n" + "#" * 85)
    print(" [SCENARIO 2/4] FEATURE & ML SCHEMA HEALING")
    print(" Failure Injected: Missing 4 required geometric features + NaN values in GLCM texture.")
    print("#" * 85)

    corrupted_features = {
        "contour_count": 2.0,
        "total_area": 950.0,
        "mean_intensity": np.nan,  # Injected NaN
        "glcm_contrast": np.nan,   # Injected NaN
        # Notice missing hu moments, solidity, aspect ratio
    }

    state2 = {
        "frame_index": 202,
        "raw_frame": clean_frame,
        "features": corrupted_features,
        "execution_log": ["[Test Harness] Injected corrupted feature vector with NaN values and missing keys."],
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    out2 = app.invoke(state2)
    print("\nExecution Trace:")
    for l in out2.get("execution_log", []):
        print(f"  > {l}")
    print("\nSelf-Healing Actions Taken:")
    for h in out2.get("healing_actions", []):
        print(f"  [HEALED] Component: {h['target_component']} | Action: {h['action_taken']}")
    print(f"Final Defect Diagnosis: {out2.get('alert', {}).get('defect_type', '').upper()} (Confidence: {round(out2.get('alert', {}).get('confidence', 0)*100, 1)}%)")
    assert out2.get("status") in ["HEALED", "COMPLETED"], "Scenario 2 failed to heal!"

    # ---------------------------------------------------------------------------------
    # SCENARIO 3: RAG & Retrieval Self-Correction (Low Relevance Search Query)
    # ---------------------------------------------------------------------------------
    print("\n" + "#" * 85)
    print(" [SCENARIO 3/4] RAG RETRIEVAL DRIFT & RELEVANCE SELF-CORRECTION")
    print(" Failure Injected: Obscure non-technical search query yielding low vector similarity.")
    print("#" * 85)

    state3 = {
        "frame_index": 203,
        "raw_frame": clean_frame,
        "features": {col: 1.0 for col in [
            "contour_count", "total_area", "mean_area", "max_area", "mean_aspect_ratio",
            "mean_extent", "mean_solidity", "mean_eccentricity", "mean_intensity",
            "std_intensity", "intensity_range", "hu_1", "hu_2", "hu_3", "hu_4", "hu_5",
            "hu_6", "hu_7", "glcm_contrast", "glcm_dissimilarity", "glcm_homogeneity"
        ]},
        "alert": {
            "frame_index": 203,
            "defect_type": "dimensional",
            "confidence": 0.95,
            "severity_score": 8.5,
            "severity_level": "CRITICAL"
        },
        "search_queries": ["extraneous non-existent manufacturing artifact query xyz99"], # Injected poor query
        "execution_log": ["[Test Harness] Injected low-relevance ambiguous search query."],
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    out3 = app.invoke(state3)
    print("\nExecution Trace:")
    for l in out3.get("execution_log", []):
        print(f"  > {l}")
    print("\nSelf-Healing Actions Taken:")
    for h in out3.get("healing_actions", []):
        print(f"  [HEALED] Component: {h['target_component']} | Action: {h['action_taken']}")
    print(f"Work Order Generated: {out3.get('work_order', {}).get('work_order_id')} (SOP Citations: {out3.get('work_order', {}).get('source_manuals')})")
    assert out3.get("status") in ["HEALED", "COMPLETED"], "Scenario 3 failed to heal!"

    # ---------------------------------------------------------------------------------
    # SCENARIO 4: Runtime Pipeline Exception Auto-Fixer (Circuit Breaker & State Patching)
    # ---------------------------------------------------------------------------------
    print("\n" + "#" * 85)
    print(" [SCENARIO 4/4] RUNTIME PIPELINE EXCEPTION AUTO-FIXER")
    print(" Failure Injected: Runtime uncaught exception simulation on worker agent.")
    print("#" * 85)

    state4 = {
        "frame_index": 204,
        "raw_frame": None, # Will trigger missing frame / exception
        "execution_log": ["[Test Harness] Injected runtime exception condition."],
        "errors": [
            {
                "component": "diagnostic_agent",
                "error_type": "runtime_exception",
                "message": "ZeroDivisionError: float division by zero in contour normalization",
                "resolved": False
            }
        ],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    out4 = app.invoke(state4)
    print("\nExecution Trace:")
    for l in out4.get("execution_log", []):
        print(f"  > {l}")
    print("\nSelf-Healing Actions Taken:")
    for h in out4.get("healing_actions", []):
        print(f"  [HEALED] Component: {h['target_component']} | Action: {h['action_taken']}")
    print(f"Recovered Alert: {out4.get('alert', {}).get('defect_type')} | Severity: {out4.get('alert', {}).get('severity_level')}")
    assert out4.get("status") in ["HEALED", "COMPLETED"], "Scenario 4 failed to heal!"

    print("\n" + "=" * 85)
    print(" ALL 4 AUTONOMOUS SELF-HEALING SCENARIOS SUCCESSFULLY VERIFIED!")
    print("=" * 85)

if __name__ == "__main__":
    run_all_healing_scenarios()
