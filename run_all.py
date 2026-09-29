import os
import sys
import time
import subprocess

# Set utf-8 encoding if possible to prevent cp1252 encoding issues on Windows
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

STEPS = [
    ("Step 1: Multi-Agent Inspection Pipeline", [sys.executable, "main.py", "--defect", "crack", "--frame", "101"]),
    ("Step 2: Autonomous Self-Healing Demo (All 4 Scenarios)", [sys.executable, "demo_self_healing.py"]),
    ("Step 3: Continuous 5-Frame Factory Conveyor Simulation", [sys.executable, "run_simulation.py"]),
    ("Step 4: Pytest Unit & Integration Test Suite", [sys.executable, "-m", "pytest", "tests/", "-v"])
]

def run_all():
    print("\n" + "=" * 90)
    print(" [SUITE] EXECUTING COMPLETE INDUSTRIAL MULTI-AGENT PLATFORM (ALL 4 WORKFLOWS)")
    print("=" * 90)

    summary = []
    total_start = time.time()

    for idx, (title, cmd) in enumerate(STEPS, 1):
        print(f"\n{'#' * 90}")
        print(f" >>> [{idx}/4] {title}")
        print(f" Command: {' '.join(cmd)}")
        print(f"{'#' * 90}\n")

        start = time.time()
        ret = subprocess.run(cmd)
        elapsed = round(time.time() - start, 2)

        status = "PASSED" if ret.returncode == 0 else "FAILED"
        summary.append((title, status, elapsed))

        if ret.returncode != 0:
            print(f"\n[Warning] {title} exited with return code {ret.returncode}")

    total_elapsed = round(time.time() - total_start, 2)

    print("\n" + "=" * 90)
    print(" [SUMMARY] EXECUTIVE EXECUTION REPORT")
    print("=" * 90)
    for title, status, elapsed in summary:
        tag = "[OK]  " if status == "PASSED" else "[FAIL]"
        print(f" {tag}  {status:<8} | {elapsed:>6.2f}s | {title}")
    print("-" * 90)
    print(f" [TIME] Total Execution Time: {total_elapsed} seconds")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    run_all()
