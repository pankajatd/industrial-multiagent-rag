import time
from src.graph import build_multiagent_graph
from src.tools.camera import VirtualCamera

def run_continuous_stream(num_frames: int = 5):
    print("=" * 80)
    print(" CONTINUOUS MULTI-AGENT FACTORY STREAM SIMULATION")
    print("=" * 80)
    
    app = build_multiagent_graph()
    cam = VirtualCamera(["normal", "scratch", "crack", "corrosion", "dimensional"])

    print(f"[Streamer] Initialized Virtual Camera feed. Processing {num_frames} frames...\n")

    for i in range(1, num_frames + 1):
        ret, frame, expected_label = cam.read()
        if not ret:
            break

        print("-" * 80)
        print(f" Frame #{i:03d} | Expected Target: {expected_label.upper()}")
        print("-" * 80)

        initial_state = {
            "frame_index": i,
            "raw_frame": frame,
            "execution_log": [f"[Streamer] Ingested live frame #{i}."],
            "errors": [],
            "healing_actions": [],
            "retry_count": 0,
            "status": "PROCESSING"
        }

        start_t = time.time()
        final_state = app.invoke(initial_state)
        elapsed = round((time.time() - start_t) * 1000, 1)

        alert = final_state.get("alert", {})
        wo = final_state.get("work_order")

        print(f"  * Status:        {final_state.get('status')}")
        print(f"  * ML Diagnosis:  {alert.get('defect_type', '').upper()} (Confidence: {round(alert.get('confidence', 0)*100, 1)}%)")
        print(f"  * Severity:      {alert.get('severity_score')} / 10.0 [{alert.get('severity_level')}]")
        
        if wo:
            print(f"  * RAG Ticket:    {wo.get('work_order_id')} (SOPs: {', '.join(wo.get('source_manuals', []))})")
            print(f"  * Signoff Gate:  {'REQUIRED' if wo.get('technician_signoff_required') else 'AUTO-APPROVED'}")
        else:
            print(f"  * RAG Ticket:    N/A (Defect within PASS / LOW tolerance)")
            
        print(f"  * Latency:       {elapsed} ms\n")

    print("=" * 80)
    print(" CONTINUOUS STREAM SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == "__main__":
    run_continuous_stream(5)
