import json
import argparse
from src.graph import build_multiagent_graph
from src.tools.camera import SyntheticIndustrialGenerator

def run_multiagent_pipeline(defect_type: str = "crack", frame_index: int = 101):
    print("=" * 80)
    print(" INDUSTRIAL MULTI-AGENT VISION & DIAGNOSTIC RAG PLATFORM (LangGraph)")
    print("=" * 80)
    print(f"[Run] Initializing simulation for target defect: '{defect_type.upper()}' (Frame #{frame_index})\n")

    # 1. Generate Synthetic Frame
    generator = SyntheticIndustrialGenerator(seed=frame_index)
    frame, label = generator.generate(defect_type)

    # 2. Build LangGraph Multi-Agent System
    print("[System] Compiling LangGraph StateGraph with Orchestrator & Self-Healing Agent...")
    app = build_multiagent_graph()

    # 3. Initialize Initial State
    initial_state = {
        "frame_index": frame_index,
        "raw_frame": frame,
        "execution_log": [f"[System] Ingested raw frame #{frame_index} from Virtual Camera."],
        "errors": [],
        "healing_actions": [],
        "retry_count": 0,
        "status": "PROCESSING"
    }

    # 4. Invoke Multi-Agent StateGraph
    print("[System] Executing Multi-Agent Graph...\n")
    final_state = app.invoke(initial_state)

    # 5. Display Multi-Agent Execution Trace
    print("-" * 80)
    print(" MULTI-AGENT EXECUTION TRACE")
    print("-" * 80)
    for log_entry in final_state.get("execution_log", []):
        print(f"  {log_entry}")

    # 6. Display Diagnostics & Severity
    alert = final_state.get("alert", {})
    print("\n" + "=" * 80)
    print(" DIAGNOSTIC AGENT REPORT")
    print("=" * 80)
    print(f" Defect Type:     {alert.get('defect_type', 'UNKNOWN').upper()}")
    print(f" Confidence:      {round(alert.get('confidence', 0.0) * 100, 2)}%")
    print(f" Severity Score:  {alert.get('severity_score', 0.0)} / 10.0")
    print(f" Severity Level:  {alert.get('severity_level', 'PASS')}")

    # 7. Display RAG Work Order (if generated)
    work_order = final_state.get("work_order")
    if work_order:
        print("\n" + "=" * 80)
        print(f" MAINTENANCE RAG WORK ORDER ({work_order.get('work_order_id')})")
        print("=" * 80)
        print(f" Severity:        {work_order.get('severity_level')}")
        print(f" Signoff Req:     {work_order.get('technician_signoff_required')}")
        print("\n Safety Directives:")
        for directive in work_order.get("safety_directives", []):
            print(f"   * {directive}")
        print("\n Repair Procedure:")
        for i, step in enumerate(work_order.get("repair_procedure", []), 1):
            print(f"   {i}. {step}")
        print(f"\n Source Manuals Cited: {', '.join(work_order.get('source_manuals', []))}")

    # 8. Display Self-Healing Actions (if any occurred)
    healing_actions = final_state.get("healing_actions", [])
    if healing_actions:
        print("\n" + "=" * 80)
        print(" AUTONOMOUS SELF-HEALING REPORT")
        print("=" * 80)
        for act in healing_actions:
            print(f"  Target: {act.get('target_component')} | Type: {act.get('failure_type')}")
            print(f"  Action: {act.get('action_taken')}")
            print(f"  Status: {act.get('status')}\n")

    print("=" * 80)
    print(f" PIPELINE FINISHED WITH FINAL STATUS: [{final_state.get('status')}]")
    print("=" * 80)
    return final_state

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Industrial Multi-Agent Vision RAG Platform")
    parser.add_argument("--defect", type=str, default="crack", choices=["normal", "scratch", "crack", "corrosion", "dimensional"], help="Target defect type")
    parser.add_argument("--frame", type=int, default=105, help="Frame index")
    args = parser.parse_args()

    run_multiagent_pipeline(defect_type=args.defect, frame_index=args.frame)
