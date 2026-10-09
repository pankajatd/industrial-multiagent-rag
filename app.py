"""
Industrial Multi-Agent Vision & Diagnostic RAG Platform — Streamlit Cloud Dashboard
====================================================================================
Autonomous 6-Agent LangGraph Swarm:
- OrchestratorAgent: Dynamic Supervisor dispatching specialized agents
- VisionAgent: Edge detection, contours, 21 geometric & GLCM texture features
- DiagnosticAgent: Random Forest defect classification & severity scoring
- MaintenanceRAGAgent: TF-IDF vector retrieval over ISO/ASTM/OSHA SOP manuals
- SelfHealingAgent: Real-time sensor deblur, CLAHE contrast boost, schema imputation
- QualityGateAgent: OSHA Lockout/Tagout audit & technician signoff enforcement
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image
import streamlit as st

from src.graph import build_multiagent_graph
from src.tools.camera import SyntheticIndustrialGenerator

# ──────────────────────────────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Industrial Multi-Agent Vision & Diagnostic RAG Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────────
# Custom Industrial Dark Styling
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 1.2rem !important;
        max-width: 100% !important;
    }
    .header-banner {
        background: linear-gradient(135deg, #0b0f19 0%, #1e1b4b 50%, #0f172a 100%);
        padding: 1.1rem 1.6rem;
        border-radius: 12px;
        margin-bottom: 1.2rem;
        color: #f3f4f6;
        border: 1px solid #374151;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .header-banner h1 {
        margin: 0;
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .header-banner p {
        margin: 0.3rem 0 0;
        opacity: 0.85;
        font-size: 0.85rem;
    }
    .stMetric {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    .work-order-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 10px;
        padding: 1.2rem;
        margin-top: 0.5rem;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .badge-critical {
        background: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid #ef4444;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-high {
        background: rgba(249, 115, 22, 0.2);
        color: #fb923c;
        border: 1px solid #f97316;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 11px;
    }
    .badge-pass {
        background: rgba(34, 197, 94, 0.2);
        color: #4ade80;
        border: 1px solid #22c55e;
        padding: 2px 8px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Caching Graph & Generator
# ──────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⚙️ Compiling LangGraph Multi-Agent StateGraph...")
def get_graph():
    return build_multiagent_graph()

multiagent_app = get_graph()

# ──────────────────────────────────────────────────────────────────────
# Header Banner
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1>🏭 Industrial Multi-Agent Vision & Diagnostic RAG Platform</h1>
            <p>Autonomous 6-Agent LangGraph Swarm • Computer Vision • Random Forest ML • Vector SOP RAG • Self-Healing Loop</p>
        </div>
        <div style="font-family: monospace; font-size: 12px; background: rgba(99, 102, 241, 0.2); border: 1px solid #6366f1; padding: 4px 12px; border-radius: 20px; color: #a5b4fc;">
            LangGraph v0.2 • OpenCV • Scikit-Learn
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────
# Sidebar Controls
# ──────────────────────────────────────────────────────────────────────
st.sidebar.markdown('<div style="font-weight:700; font-size:14px; color:#e2e8f0; margin-bottom:8px;">⚙️ Conveyor & Camera Controls</div>', unsafe_allow_html=True)

defect_choice = st.sidebar.selectbox(
    "Target Surface Defect",
    ["crack", "corrosion", "dimensional", "scratch", "normal"],
    index=0,
    format_func=lambda d: {
        "crack": "⚡ Structural Crack (Critical)",
        "corrosion": "🧪 Surface Oxidation / Corrosion (High)",
        "dimensional": "📐 Dimensional Flaw (Medium)",
        "scratch": "✏️ Surface Scratch (Low)",
        "normal": "✨ Pristine / Normal Surface (Pass)"
    }.get(d, d),
    key="defect_choice"
)

frame_seed = st.sidebar.number_input(
    "Virtual Camera Frame ID / Seed",
    min_value=1,
    max_value=9999,
    value=101,
    step=1,
    key="frame_seed"
)

st.sidebar.markdown("---")
st.sidebar.markdown('<div style="font-weight:700; font-size:14px; color:#e2e8f0; margin-bottom:8px;">🔧 Autonomous Self-Healing Simulation</div>', unsafe_allow_html=True)

failure_injection = st.sidebar.selectbox(
    "Simulate Optical Sensor / Data Flaw",
    ["none", "blur", "underexposed", "overexposed", "corrupt_schema"],
    format_func=lambda f: {
        "none": "🟢 None (Clean Sensor Capture)",
        "blur": "💨 Heavy Defocus Blur (Sensor Drift)",
        "underexposed": "🌑 Severe Underexposure (Strobe Failure)",
        "overexposed": "☀️ Blinding Specular Glare (Overexposure)",
        "corrupt_schema": "🧬 Corrupted Feature Schema (NaN & Missing Attributes)"
    }.get(f, f),
    key="failure_injection"
)

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🚀 Run Multi-Agent Diagnostic Pipeline", type="primary", use_container_width=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**6-Agent LangGraph Swarm:**
1. 🧭 **OrchestratorAgent**: Supervisor Router
2. 👁️ **VisionAgent**: 21 CV & Texture Features
3. 🔬 **DiagnosticAgent**: Random Forest Classifier
4. 📋 **MaintenanceRAGAgent**: Vector SOP Retrieval
5. 🛡️ **QualityGateAgent**: OSHA Compliance Audit
6. 🔧 **SelfHealingAgent**: Closed-Loop Recovery
""")

# ──────────────────────────────────────────────────────────────────────
# Frame Generation & Failure Injection
# ──────────────────────────────────────────────────────────────────────
generator = SyntheticIndustrialGenerator(seed=int(frame_seed))
clean_frame, _ = generator.generate(defect_choice)

input_frame = clean_frame.copy()
simulated_note = ""

if failure_injection == "blur":
    input_frame = generator.inject_blur(clean_frame, ksize=25)
    simulated_note = "Simulated severe camera defocus blur (ksize=25)."
elif failure_injection == "underexposed":
    input_frame = generator.inject_underexposure(clean_frame, factor=0.18)
    simulated_note = "Simulated optical strobe illumination blackout (factor=0.18)."
elif failure_injection == "overexposed":
    input_frame = generator.inject_overexposure(clean_frame, offset=140)
    simulated_note = "Simulated extreme specular glare overexposure (+140 offset)."

# ──────────────────────────────────────────────────────────────────────
# Run Multi-Agent StateGraph
# ──────────────────────────────────────────────────────────────────────
initial_log = [f"[System] Ingested raw frame #{frame_seed} from Virtual Camera."]
if simulated_note:
    initial_log.append(f"[Failure Injection] {simulated_note}")

initial_state = {
    "frame_index": int(frame_seed),
    "raw_frame": input_frame,
    "execution_log": initial_log,
    "errors": [],
    "healing_actions": [],
    "retry_count": 0,
    "status": "PROCESSING"
}

if failure_injection == "corrupt_schema":
    # Inject corrupted schema into initial state
    initial_state["errors"].append("Schema validation failure: 4 missing attributes + NaNs injected.")

with st.spinner("🤖 Autonomous Multi-Agent Swarm executing..."):
    final_state = multiagent_app.invoke(initial_state)

# Extract State Artifacts
alert = final_state.get("alert", {})
defect_detected = alert.get("defect_type", "UNKNOWN").upper()
confidence = alert.get("confidence", 0.0)
severity_score = alert.get("severity_score", 0.0)
severity_level = alert.get("severity_level", "NORMAL")
healing_actions = final_state.get("healing_actions", [])
work_order = final_state.get("work_order")
features = final_state.get("features", {})
execution_log = final_state.get("execution_log", [])
processed_frame = final_state.get("processed_frame")
mask = final_state.get("mask")

# ──────────────────────────────────────────────────────────────────────
# Top KPI Metric Cards Bar
# ──────────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

with k1:
    badge_label = "✅ PASS" if defect_detected == "NORMAL" else f"⚠️ {defect_detected}"
    st.metric("Defect Classification", badge_label, f"Target: {defect_choice.upper()}")

with k2:
    st.metric("Model Confidence", f"{confidence * 100:.1f}%", "Random Forest Ensemble")

with k3:
    st.metric("Severity Score", f"{severity_score:.1f} / 10.0", f"Level: {severity_level}")

with k4:
    if healing_actions:
        st.metric("Self-Healing Engine", f"HEALED ({len(healing_actions)} Fix{'es' if len(healing_actions) > 1 else ''})", "Closed-Loop Recovery")
    else:
        st.metric("Self-Healing Engine", "PASS (Nominal)", "Zero Anomalies Detected")

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Dual-Viewport Camera Feed
# ──────────────────────────────────────────────────────────────────────
st.markdown("### 📷 Inspection Viewport — Optical Sensor Capture vs Vision Segmentation")
col_cam1, col_cam2 = st.columns(2, gap="medium")

with col_cam1:
    st.markdown("**1. Ingested Optical Camera Frame** *(Virtual Camera Sensor)*")
    disp_in = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
    st.image(disp_in, use_container_width=True)

with col_cam2:
    st.markdown("**2. Computer Vision Feature Analysis & Defect Overlay**")
    if processed_frame is not None:
        disp_proc = processed_frame.copy()
        # Overlay red mask if available
        if mask is not None and np.sum(mask) > 0:
            overlay = disp_proc.copy()
            overlay[mask > 0] = [0, 0, 255]
            disp_proc = cv2.addWeighted(disp_proc, 0.7, overlay, 0.3, 0)
        disp_out = cv2.cvtColor(disp_proc, cv2.COLOR_BGR2RGB)
    else:
        disp_out = disp_in
    st.image(disp_out, use_container_width=True)

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────
# Multi-Agent Telemetry & Decision Tabs
# ──────────────────────────────────────────────────────────────────────
tab_wo, tab_heal, tab_feat, tab_graph, tab_logs = st.tabs([
    "📋 Maintenance RAG Work Order",
    "🔧 Autonomous Self-Healing Audit",
    "📊 Vision & Geometry Telemetry",
    "🧠 Multi-Agent LangGraph Swarm",
    "📜 Live Execution Trace"
])

with tab_wo:
    if work_order:
        wo_id = work_order.get("work_order_id", "WO-UNKNOWN")
        signoff = work_order.get("technician_signoff_required", False)
        
        st.markdown(f"""
        <div class="work-order-card">
            <div style="display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #374151; padding-bottom: 8px; margin-bottom: 12px;">
                <h3 style="margin: 0; color: #60a5fa; font-family: monospace;">📋 WORK ORDER: {wo_id}</h3>
                <div>
                    <span class="{'badge-critical' if severity_level == 'CRITICAL' else 'badge-high' if severity_level == 'HIGH' else 'badge-pass'}">
                        {severity_level}
                    </span>
                    <span style="margin-left: 8px; font-size: 11px; padding: 2px 8px; border-radius: 9999px; background: rgba(147, 51, 234, 0.2); color: #c084fc; border: 1px solid #9333ea;">
                        Signoff Required: {'YES ✍️' if signoff else 'NO'}
                    </span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 🛡️ OSHA & Safety Directives")
        for directive in work_order.get("safety_directives", []):
            st.markdown(f"- ⚠️ **{directive}**")
            
        st.markdown("#### 🛠️ Authorized Repair Procedure")
        for i, step in enumerate(work_order.get("repair_procedure", []), 1):
            st.markdown(f"{i}. {step}")
            
        st.markdown("#### 📚 Source Standard Operating Procedures (SOP Citations)")
        manuals = work_order.get("source_manuals", [])
        if manuals:
            for m in manuals:
                st.code(f"📄 Cited Reference: {m}", language="markdown")
        else:
            st.info("No SOP citation required for pristine state.")
    else:
        st.success("✨ **Surface is Nominal / Defect Free.** No maintenance work order required.")

with tab_heal:
    if healing_actions:
        st.markdown("### 🛡️ Closed-Loop Self-Healing Interventions Applied")
        for i, act in enumerate(healing_actions, 1):
            with st.container():
                st.markdown(f"""
                <div style="background: #1e1b4b; border: 1px solid #6366f1; border-radius: 8px; padding: 12px; margin-bottom: 10px;">
                    <div style="color: #a5b4fc; font-weight: 700; font-size: 13px;">Fix #{i}: {act.get('target_component', 'COMPONENT').upper()} — {act.get('failure_type', 'ANOMALY')}</div>
                    <div style="color: #e0e7ff; font-size: 13px; margin-top: 4px;">🔧 <b>Action Taken:</b> {act.get('action_taken')}</div>
                    <div style="color: #4ade80; font-size: 12px; margin-top: 2px;">Status: {act.get('status', 'RESOLVED')} ✅</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("✨ System operated in zero-degradation state. No self-healing actions were triggered.")

with tab_feat:
    st.markdown("### 🔬 21 Extracted Computer Vision & Texture Features")
    if features:
        feat_cols = st.columns(3)
        feat_list = list(features.items())
        chunk = (len(feat_list) + 2) // 3
        for col_idx in range(3):
            with feat_cols[col_idx]:
                for k, v in feat_list[col_idx * chunk : (col_idx + 1) * chunk]:
                    val_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                    st.text(f"• {k}: {val_str}")
    else:
        st.info("No geometric features extracted.")

with tab_graph:
    st.markdown("""
```mermaid
graph TD
    CAM["📹 Virtual Camera Sensor Ingestion"] --> ORCH["🧭 Orchestrator Agent (Supervisor)"]
    ORCH -->|Dispatch Frame| VIS["👁️ Vision Agent (21 Edge/Texture Features)"]
    VIS -->|Features Extracted| DIAG["🔬 Diagnostic Agent (Random Forest ML)"]
    DIAG -->|Defect Detected| RAG["📋 Maintenance RAG Agent (TF-IDF Vector SOPs)"]
    RAG -->|Formulated Work Order| QG["🛡️ Quality Gate Agent (OSHA Signoff Audit)"]
    QG -->|Approved| FIN["✅ Final Certified Dispatch"]
    
    VIS -.->|Optical Degradation / Sensor Flaw| HEAL["🔧 Autonomous Self-Healing Agent"]
    DIAG -.->|Schema Corruption / NaN Attributes| HEAL
    HEAL -.->|Deblurred / Contrast Boosted / Imputed| ORCH
```
    """)

with tab_logs:
    st.markdown("### 📜 Real-Time Agent Execution Trace")
    for log in execution_log:
        if "CRITICAL" in log or "Error" in log:
            st.error(log)
        elif "HEALED" in log or "Autonomous" in log:
            st.warning(log)
        elif "success" in log or "Verified" in log:
            st.success(log)
        else:
            st.code(log, language="bash")

# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div style="text-align:center; opacity:0.6; font-size:0.8rem;">'
    '🏭 Industrial Multi-Agent Vision & Diagnostic RAG Platform • '
    'Powered by LangGraph, OpenCV, Scikit-Learn & Streamlit Cloud'
    '</div>',
    unsafe_allow_html=True,
)
