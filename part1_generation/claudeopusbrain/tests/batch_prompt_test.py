# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System

#!/usr/bin/env python3
"""
Batch prompt test: try different prompts for same image,
keep the best result based on validation scores.

Run:
    conda run -n wan2.2 python3 batch_prompt_test.py
"""

import subprocess
import json
import re
import shutil
from pathlib import Path
from datetime import datetime

IMAGE = "/home/humanoid-isr/LTX-2/web/uploads/Screenshot_from_2026-02-06_13-44-44_20260206_134452.png"
PIPELINE = "/mnt/Thesis/JeffrinSam/Part1/AgentLLM/Manipulation/manipulation_pipeline.py"
OUTPUT_DIR = Path("/mnt/Thesis/JeffrinSam/Part1/Claudeopusbrain/tests/batch_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Different prompts to test — from simple to detailed, left/right arm, 5s and 10s
PROMPTS = [
    {
        "name": "right_arm_basic",
        "desc": "Right arm, basic grasp",
        "prompt": "G1 humanoid robot right arm reaches forward and down toward the ball, hand opens wide, fingers close tightly around the ball, arm lifts ball up smoothly off the table",
        "frames": 77,   # ~5 sec
    },
    {
        "name": "left_arm_basic",
        "desc": "Left arm, basic grasp",
        "prompt": "G1 humanoid robot left arm extends toward the ball on the table, hand descends, fingers curl around the ball grasping it firmly, lifts ball up to chest height",
        "frames": 77,
    },
    {
        "name": "right_arm_detailed",
        "desc": "Right arm, detailed temporal sequence",
        "prompt": "Unitree G1 humanoid robot manipulation sequence: initial pose standing upright, right shoulder joint rotates forward, elbow extends arm downward toward ball, wrist pronates, five fingers spread open above ball, fingers close wrapping firmly around ball, forearm lifts upward raising ball 30cm off table surface, final pose holding ball at waist level",
        "frames": 77,
    },
    {
        "name": "bimanual_transfer",
        "desc": "Bimanual: right picks, transfer to left",
        "prompt": "G1 humanoid bimanual manipulation: right hand grasps ball from table with pinch grip, lifts ball upward, left hand moves to meet it, ball transferred from right to left hand, left hand closes around ball",
        "frames": 100,  # slightly longer
    },
    {
        "name": "right_arm_10sec",
        "desc": "Right arm, 10 second video",
        "prompt": "Unitree G1 humanoid robot: standing in front of table with ball. Right arm slowly extends toward ball. Hand opens. Fingers contact ball surface. Fingers curl closed grasping ball. Wrist rotates. Arm raises ball up from table. Ball held securely at chest height. Robot pauses briefly. Places ball back on table gently.",
        "frames": 161,  # ~10 sec at 16fps
    },
]


def extract_scores_from_json(json_path: Path) -> dict:
    """Extract scores robustly — handles XML parse failures by reading raw_response."""
    try:
        with open(json_path) as f:
            data = json.load(f)

        # Check if scores are valid (not all 50 = parsing error)
        components = data.get("components", [])
        if components and all(c.get("score") == 50 for c in components):
            # XML failed, extract from raw_response
            raw = data.get("raw_response", "")
            scores = re.findall(r'<score>(\d+)</score>', raw)
            if len(scores) >= 3:
                return {
                    "adherence": int(scores[0]),
                    "physics": int(scores[1]),
                    "quality": int(scores[2]),
                    "avg": sum(int(s) for s in scores[:3]) / 3,
                    "pass": sum(int(s) for s in scores[:3]) / 3 >= 80,
                    "raw_ok": True,
                }

        # Normal parse
        scores = {c["name"].split()[0].lower(): c["score"] for c in components}
        avg = sum(c["score"] for c in components) / len(components) if components else 0
        return {
            "adherence": scores.get("prompt", 0),
            "physics": scores.get("physical", 0),
            "quality": scores.get("visual", 0),
            "avg": avg,
            "pass": data.get("pass", False),
            "raw_ok": False,
        }
    except Exception as e:
        return {"adherence": 0, "physics": 0, "quality": 0, "avg": 0, "pass": False, "error": str(e)}


def run_pipeline(prompt: str, output_mp4: Path, frames: int) -> bool:
    """Run manipulation pipeline."""
    cmd = [
        "python3", PIPELINE,
        "--task", "g1",
        "--prompt", prompt,
        "--image", IMAGE,
        "--output", str(output_mp4),
        "--cosmos-frames", str(frames),
    ]
    print(f"   Running: frames={frames}, prompt={prompt[:60]}...")
    result = subprocess.run(cmd, cwd=str(Path(PIPELINE).parent))
    return result.returncode == 0


def main():
    print("=" * 70)
    print("BATCH PROMPT TEST — G1 Humanoid Ball Pickup")
    print(f"Image: {IMAGE}")
    print(f"Testing {len(PROMPTS)} prompt variations")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    results = []

    for i, p in enumerate(PROMPTS, 1):
        print(f"\n[{i}/{len(PROMPTS)}] {p['name']} — {p['desc']}")
        print("-" * 50)

        output_mp4 = OUTPUT_DIR / f"{p['name']}.mp4"
        val_json = Path(PIPELINE).parent / "outputs" / f"validation_{p['name']}.json"

        # Run pipeline
        success = run_pipeline(p["prompt"], output_mp4, p["frames"])

        if not success or not output_mp4.exists():
            print(f"   FAILED — video not generated")
            results.append({**p, "scores": {"avg": 0, "error": "generation failed"}})
            continue

        print(f"   Video: {output_mp4} ({output_mp4.stat().st_size // 1024}K)")

        # Extract scores
        scores = extract_scores_from_json(val_json) if val_json.exists() else {"avg": 0, "error": "no validation"}
        print(f"   Scores: adherence={scores.get('adherence',0)} physics={scores.get('physics',0)} quality={scores.get('quality',0)} avg={scores.get('avg',0):.1f}")

        results.append({**p, "scores": scores, "video": str(output_mp4)})

    # Find best result
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    best = max(results, key=lambda r: r["scores"].get("avg", 0))

    for r in results:
        avg = r["scores"].get("avg", 0)
        marker = " ← BEST" if r == best else ""
        print(f"  {r['name']:25s}  avg={avg:5.1f}  {r['desc']}{marker}")

    # Copy best to tests/best_result.mp4
    if best.get("video") and Path(best["video"]).exists():
        best_path = OUTPUT_DIR.parent / "best_g1_pickup.mp4"
        shutil.copy2(best["video"], best_path)
        print(f"\nBest video saved: {best_path}")
        print(f"Best prompt: {best['prompt']}")
        print(f"Best scores: {best['scores']}")

    # Save full results
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "image": IMAGE,
            "best": best["name"],
            "results": [{k: v for k, v in r.items() if k != "prompt"} | {"prompt_preview": r["prompt"][:80]} for r in results]
        }, f, indent=2)
    print(f"Summary: {summary_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
