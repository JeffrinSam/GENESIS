#!/usr/bin/env python3
"""
Generate LTX-2 keyframe interpolation videos for 3-keyframe half-C trajectory.
Uses: 00_start_fpv -> 03_curving_fpv -> 08_end_fpv (2 pairs)
Saves to folder "2" inside continuous_trajectory.
"""

import os
import json
import time
import shutil
import subprocess
import requests
from pathlib import Path

API_URL = "http://192.168.50.253:5001"
UPLOAD_URL = f"{API_URL}/upload"
GENERATE_URL = f"{API_URL}/generate/keyframe_interpolation"

KEYFRAMES_DIR = Path(__file__).parent / "keyframes"
OUTPUT_DIR = Path(__file__).parent / "2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PAIRS = [
    ("00_start_fpv.png", "03_curving_fpv.png",
     "A mobile robot moves forward through a hospital corridor, curving gently to the right. A vending machine on the right side grows closer as the robot approaches. Ceiling lights pass overhead smoothly. Indoor hospital hallway, first person view."),
    ("03_curving_fpv.png", "08_end_fpv.png",
     "A mobile robot completes a smooth right curve in a hospital corridor, arriving in front of a large vending machine. The vending machine shifts from center to fill the left side of the view. Smooth forward approach, hospital interior, first person perspective."),
]


def upload_image(filepath):
    with open(filepath, 'rb') as f:
        resp = requests.post(UPLOAD_URL, files={'image': f})
    resp.raise_for_status()
    data = resp.json()
    if not data.get('success'):
        raise RuntimeError(f"Upload failed: {data}")
    return data['path']


def generate_interpolation(start_path, end_path, prompt, output_name, pair_idx):
    payload = {
        "start_image_path": start_path,
        "end_image_path": end_path,
        "prompt": prompt,
        "resolution": "512x512",
        "num_frames": 81,
        "cfg_guidance_scale": 3.5,
        "num_inference_steps": 40,
        "start_strength": 1.0,
        "end_strength": 1.0,
        "seed": 100 + pair_idx,
    }

    print(f"  Generating... (this may take 1-2 minutes)")
    resp = requests.post(GENERATE_URL, json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()

    if not data.get('success'):
        raise RuntimeError(f"Generation failed: {data.get('error', 'Unknown error')}")

    video_url = data['video_url']
    video_resp = requests.get(f"{API_URL}{video_url}")
    video_resp.raise_for_status()

    output_path = OUTPUT_DIR / output_name
    with open(output_path, 'wb') as f:
        f.write(video_resp.content)

    gen_time = data.get('generation_time', 0)
    print(f"  Done in {gen_time:.1f}s -> {output_path.name}")
    return output_path


def concatenate_videos(video_paths, output_path):
    list_file = OUTPUT_DIR / "concat_list.txt"
    with open(list_file, 'w') as f:
        for vp in video_paths:
            f.write(f"file '{vp}'\n")

    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(list_file),
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p",
        str(output_path)
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    list_file.unlink()
    print(f"\nFinal video: {output_path} ({output_path.stat().st_size / 1024 / 1024:.1f}MB)")


def main():
    print("=" * 60)
    print("  LTX-2 Keyframe Interpolation — Folder 2 (3 keyframes)")
    print("=" * 60)

    try:
        resp = requests.get(f"{API_URL}/status", timeout=5)
        status = resp.json()
        print(f"Server: OK | GPU: {status.get('gpu', {}).get('name', 'unknown')}")
        if status.get('busy'):
            print("WARNING: Server is busy!")
            return
    except Exception as e:
        print(f"ERROR: Cannot reach server at {API_URL}: {e}")
        return

    generated_videos = []

    for i, (start_file, end_file, prompt) in enumerate(PAIRS):
        print(f"\n--- Pair {i+1}/{len(PAIRS)}: {start_file} -> {end_file} ---")

        start_local = KEYFRAMES_DIR / start_file
        end_local = KEYFRAMES_DIR / end_file

        if not start_local.exists() or not end_local.exists():
            print(f"  SKIP: Missing keyframe files")
            continue

        print(f"  Uploading {start_file}...")
        start_server = upload_image(start_local)
        print(f"  Uploading {end_file}...")
        end_server = upload_image(end_local)

        output_name = f"pair_{i:02d}.mp4"
        try:
            video_path = generate_interpolation(start_server, end_server, prompt, output_name, i)
            generated_videos.append(video_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if i < len(PAIRS) - 1:
            print("  Waiting 5s for VRAM cleanup...")
            time.sleep(5)

    if len(generated_videos) >= 2:
        print(f"\n--- Concatenating {len(generated_videos)} clips ---")
        final_path = OUTPUT_DIR / "half_c_combined.mp4"
        concatenate_videos(generated_videos, final_path)
    elif len(generated_videos) == 1:
        shutil.copy(generated_videos[0], OUTPUT_DIR / "half_c_combined.mp4")

    meta = {
        "pairs": [{"start": p[0], "end": p[1], "prompt": p[2]} for p in PAIRS],
        "num_clips": len(generated_videos),
        "clips": [str(v) for v in generated_videos],
    }
    with open(OUTPUT_DIR / "generation_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\nAll done! Output in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
