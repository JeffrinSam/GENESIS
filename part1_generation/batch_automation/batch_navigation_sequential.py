#!/usr/bin/env python3
"""
GPU-optimized Navigation Batch Runner
Sequential load/unload pattern per image per stage:
  1. Load Qwen3.5 -> enhance prompt(s) -> UNLOAD
  2. Load WAN 2.2 -> generate video(s) -> UNLOAD
  3. Load Cosmos-Reason2 -> validate video(s) -> UNLOAD
  4. Next image, repeat

Designed for low-memory GPUs where only one heavy model should be active at a time.
"""

import argparse
import glob
import json
import logging
import re
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import os

_PART1_DIR = Path(__file__).resolve().parents[1]  # part1_generation/
QWEN_EXTENDERS = Path(os.getenv("QWEN_ROOT", str(_PART1_DIR / "qwen_prompt_expansion"))) / "prompt_extenders"
WAN_GENERATE = Path(os.getenv("WAN_ROOT", "/opt/wan2.2")) / "generate.py"
WAN_CKPT_DIR = Path(os.getenv("WAN_ROOT", "/opt/wan2.2")) / "Wan2.2-TI2V-5B"
VALIDATOR = _PART1_DIR / "agentllm" / "Navigation" / "video_validator.py"
REASON2_VENV_PY = Path(os.getenv("COSMOS_REASON2_ROOT", "/opt/cosmos-reason2")) / ".venv" / "bin" / "python"
OUTPUT_BASE = Path(__file__).parent / "results" / "navigation"
DEFAULT_QWEN_ENV = "wan2.2"
DEFAULT_WAN_ENV = "wan2.2"
DEFAULT_NAV_DIR = os.getenv("NAV_DATA_DIR", str(Path.home() / "navigation_data"))
CONDA_ENVS_DIR = Path(os.getenv("CONDA_ENVS_DIR", str(Path.home() / "anaconda3" / "envs")))

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")
log = logging.getLogger("batch_navigation_sequential")

DEFAULT_HUMANOID_SYSTEM_PROMPT = (
    "You are an expert embodied cinematographer for Unitree G1 first-person navigation prompts for WAN 2.2 TI2V. "
    "The camera is the robot's eyes (first-person only). Never describe the robot from outside. "
    "Primary objective: produce strong continuous ego-motion, not a static shot. "
    "Write one continuous 5-second shot in 90-120 words. "
    "Follow this structure: "
    "(1) start state with eye-level head camera and heading, "
    "(2) clear forward translation with speed (slow/steady/brisk) and distance progression, "
    "(3) obstacle-avoidance maneuver (turn/veer/realign), "
    "(4) ending alignment or stop. "
    "Use motion verbs: dolly forward, tracking forward, pan/yaw left or right, gentle deceleration. "
    "Include parallax and floor flow cues so movement is obvious frame-to-frame. "
    "Preserve user command intent exactly (forward, turn, align, stop, avoid). "
    "Static-world rule: obstacles and scene objects stay fixed in world coordinates; avoid obstacles by path change only, no pushing or dragging objects. "
    "EGO-ONLY RULE: no independent actor motion. Do not animate people, robots, or objects independently of camera motion. "
    "Do not introduce a new human/robot entering the frame. If a person/robot exists in the image, keep it static background context unless the user explicitly commands interaction. "
    "Keep eye-level camera height approximately constant with only subtle walking bob. "
    "No cuts, no time skip, no teleportation, no zoom, no aerial/third-person view. "
    "Output only the final prompt paragraph."
)

DEFAULT_STATIC_WORLD_NEGATIVE_PROMPT = (
    "dynamic moving objects, furniture sliding, walls moving, floor warping, object drift, object morphing, scene breathing, "
    "jelly artifacts, geometry deformation, non-rigid background, camera zoom, focal-length shift, scene cut, jump cut, teleportation, "
    "third person view, aerial view, floating camera, collision, pushing obstacles, dragging objects, "
    "person walking into frame, human entering scene, external robot entering frame, humanoid moving independently, "
    "independent actor motion, object moving on its own"
)


def filename_to_prompt(path: Path) -> str:
    name = path.name
    if "." in name:
        name = name.rsplit(".", 1)[0]
    prompt = name.replace("_", " ").replace("-", " ").replace(",", " ")
    return " ".join(prompt.split())


def to_humanoid_fps_prompt(raw_prompt: str) -> str:
    """Keep the user command compact; system prompt enforces cinematography and constraints."""
    clean = " ".join(raw_prompt.split())
    return f"First-person head-camera navigation command (ego-motion only): {clean}"


def env_python(env_name: str) -> Path:
    return CONDA_ENVS_DIR / env_name / "bin" / "python3"


def write_simple_yaml(path: Path, data: dict):
    lines = []
    for k, v in data.items():
        if isinstance(v, bool):
            val = "true" if v else "false"
        elif v is None:
            val = "null"
        elif isinstance(v, (int, float)):
            val = str(v)
        else:
            text = str(v).replace("'", "''")
            val = f"'{text}'"
        lines.append(f"{k}: {val}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_sample_json(
    sample_json_path: Path,
    name: str,
    prompt: str,
    seed: int,
    guidance: float,
    image_path: str,
    output_video: str,
    negative_prompt: Optional[str],
    num_frames: int,
    steps: int,
    size: str,
    validation_path: Optional[str] = None,
):
    payload = {
        "name": name,
        "prompt_path": None,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "seed": seed,
        "guidance": guidance,
        "inference_type": "image2video",
        "input_path": image_path,
        "output_path": output_video,
        "resolution": size,
        "num_output_frames": num_frames,
        "num_steps": steps,
        "validation_path": validation_path,
    }
    sample_json_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _extract_enhanced_prompt(prompt_file: Path, fallback: str) -> str:
    if not prompt_file.exists():
        return fallback
    text = prompt_file.read_text(encoding="utf-8", errors="ignore")
    if "====" not in text:
        return " ".join(text.split()) or fallback

    lines = text.splitlines()
    capture = False
    chunks = []
    for line in lines:
        stripped = line.strip()
        if stripped and set(stripped) == {"="}:
            if not capture:
                capture = True
                continue
            break
        if capture and stripped and "Negative Prompt" not in stripped:
            chunks.append(stripped)
    enhanced = " ".join(chunks).strip()
    return enhanced or fallback


def _extract_negative_prompt(prompt_file: Path) -> Optional[str]:
    if not prompt_file.exists():
        return None
    text = prompt_file.read_text(encoding="utf-8", errors="ignore")
    marker = "Negative Prompt:"
    idx = text.find(marker)
    if idx == -1:
        return None
    tail = text[idx + len(marker):].strip()
    # Keep first non-empty lines as one sentence block.
    lines = [ln.strip() for ln in tail.splitlines() if ln.strip() and set(ln.strip()) != {"="}]
    if not lines:
        return None
    return " ".join(lines).strip()


def constrain_wan_prompt(text: str, min_words: int = 90, max_words: int = 120) -> str:
    """Normalize generated text into one WAN-friendly continuous shot prompt."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    # Remove markdown-like artifacts that sometimes leak from instruction-following.
    cleaned = cleaned.replace("**", "").replace("`", "")
    words = cleaned.split()
    if not words:
        return text

    # Trim to sentence boundary while keeping <= max_words.
    if len(words) > max_words:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned)
        kept = []
        total = 0
        for sentence in sentences:
            sw = sentence.split()
            if not sw:
                continue
            if total + len(sw) > max_words:
                break
            kept.append(sentence)
            total += len(sw)
        if kept:
            cleaned = " ".join(kept).strip()
        else:
            cleaned = " ".join(words[:max_words]).strip()

    # If too short, add a neutral motion constraint sentence (no new scene objects).
    if len(cleaned.split()) < min_words:
        tail = (
            " Maintain strict first-person eye-level viewpoint with smooth continuous motion, "
            "stable horizon, gentle turns, clear forward progression, and no zoom or scene cuts."
        )
        cleaned = (cleaned + tail).strip()
        if len(cleaned.split()) > max_words:
            cleaned = " ".join(cleaned.split()[:max_words]).strip()

    return cleaned


def _trim_to_words_with_sentence_preference(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text.strip()
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    kept = []
    total = 0
    for sentence in sentences:
        sw = sentence.split()
        if not sw:
            continue
        if total + len(sw) > limit:
            break
        kept.append(sentence.strip())
        total += len(sw)
    if kept:
        return " ".join(kept).strip()
    return " ".join(words[:limit]).strip()


def sanitize_ego_only_wording(text: str) -> str:
    """Reduce external third-person phrasing; keep camera-centric language."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    replacements = [
        (r"\bthe robot\b", "the camera"),
        (r"\brobot's\b", "camera's"),
        (r"\brobot viewpoint\b", "camera viewpoint"),
        (r"\bfrom the robot\b", "from the camera"),
        (r"\brobot moves\b", "camera moves"),
        (r"\brobot navigates\b", "camera navigates"),
    ]
    for pattern, repl in replacements:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)
    return cleaned


def enforce_ground_constraints(text: str, min_words: int = 90, max_words: int = 120) -> str:
    """Finalize ground prompt so key safety constraints are always preserved."""
    base = sanitize_ego_only_wording(re.sub(r"\s+", " ", text).strip())
    constraint_suffix = (
        "Maintain constant eye-level camera height with stabilized horizon, minimal pitch, and no vertical rise or drop. "
        "Keep all scene objects fixed in world coordinates; apparent motion comes from ego-motion and parallax only. "
        "Show continuous forward ego-motion with clear floor flow and parallax. "
        "Avoid obstacles by changing path and heading only, without moving, pushing, or dragging any object. "
        "No independent human/robot/object motion and no new actor entering the frame."
    )
    suffix_words = len(constraint_suffix.split())
    body_limit = max(1, max_words - suffix_words)
    body = _trim_to_words_with_sentence_preference(base, body_limit)
    merged = f"{body} {constraint_suffix}".strip()

    # If merged prompt is still short, add a neutral motion sentence at the front.
    if len(merged.split()) < min_words:
        preface = "Single continuous first-person humanoid navigation shot with smooth forward progression."
        merged = f"{preface} {merged}".strip()
        merged = " ".join(merged.split())
        if len(merged.split()) > max_words:
            merged = " ".join(merged.split()[:max_words]).strip()
    return merged


def enhance_prompt(
    task: str,
    prompt: str,
    image_path: str,
    qwen_env: str,
    qwen_system_prompt: Optional[str] = None,
) -> tuple[str, Optional[str]]:
    task_map = {
        "drone": QWEN_EXTENDERS / "wan22" / "prompt_extender_drone.py",
        "ground": QWEN_EXTENDERS / "wan22" / "prompt_extender_ground_robot.py",
        "g1_nav": QWEN_EXTENDERS / "wan22" / "prompt_extender_unitree_g1_nav.py",
    }
    extender = task_map.get(task)
    if not extender or not extender.exists():
        log.warning("Extender not found for %s. Using filename prompt.", task)
        return prompt, DEFAULT_STATIC_WORLD_NEGATIVE_PROMPT if task in ("ground", "g1_nav") else None

    out_base = f"nav_{task}_{uuid.uuid4().hex[:8]}"
    outputs_dir = QWEN_EXTENDERS / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Use Qwen3.5 dedicated venv (needs transformers >= 5.0 for qwen3_5 model type)
    qwen_python = ROOT / "Qwen3-VL" / ".venv" / "bin" / "python"
    if not qwen_python.exists():
        raise RuntimeError(f"Qwen3.5 venv python not found: {qwen_python}")

    cmd = [str(qwen_python), str(extender), "--prompt", prompt, "--image", image_path, "--output", out_base]
    if qwen_system_prompt:
        cmd.extend(["--system_prompt", qwen_system_prompt])

    log.info("[STAGE 1/3] Loading Qwen3.5 (%s)...", qwen_env)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.warning("  Extender failed, fallback to filename prompt.")
        if res.stderr:
            log.warning("  Extender stderr: %s", res.stderr[:200])
        return prompt, DEFAULT_STATIC_WORLD_NEGATIVE_PROMPT if task in ("ground", "g1_nav") else None

    prompt_file = outputs_dir / f"{out_base}_prompt.txt"
    enhanced_raw = _extract_enhanced_prompt(prompt_file, prompt)
    neg_raw = _extract_negative_prompt(prompt_file)
    enhanced = constrain_wan_prompt(enhanced_raw, min_words=90, max_words=120)
    neg = neg_raw.strip() if neg_raw else None
    if not neg and task in ("ground", "g1_nav"):
        neg = DEFAULT_STATIC_WORLD_NEGATIVE_PROMPT
    log.info("  Enhanced prompt words: %d (raw: %d)", len(enhanced.split()), len(enhanced_raw.split()))
    log.info("[STAGE 1/3] Qwen3.5 UNLOADED")
    return enhanced, neg


def generate_videos(
    image_path: str,
    prompt: str,
    negative_prompt: Optional[str],
    outputs: list[tuple[str, int, int]],
    wan_env: str,
    steps: int,
    guidance: float,
    size: str,
):
    """Generate videos with per-sample frame counts.

    Args:
        outputs: List of (video_path, seed, frame_count) tuples.
                 frame_count varies per sample (e.g. 121 for 5s, 242 for 10s).
    """
    if not WAN_GENERATE.exists():
        raise RuntimeError(f"WAN generate script missing: {WAN_GENERATE}")
    if not WAN_CKPT_DIR.exists():
        raise RuntimeError(f"WAN checkpoint missing: {WAN_CKPT_DIR}")

    wan_python = env_python(wan_env)
    if not wan_python.exists():
        raise RuntimeError(f"WAN env python not found: {wan_python}")

    log.info("[STAGE 2/3] Loading WAN 2.2 (%s)...", wan_env)
    for out_video, seed, frames in outputs:
        duration_s = frames / 24  # WAN 2.2: 121f=~5s, 242f=~10s
        Path(out_video).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(wan_python),
            str(WAN_GENERATE),
            "--task",
            "ti2v-5B",
            "--ckpt_dir",
            str(WAN_CKPT_DIR),
            "--prompt",
            prompt,
            "--image",
            image_path,
            "--size",
            size,
            "--frame_num",
            str(frames),
            "--sample_steps",
            str(steps),
            "--sample_guide_scale",
            str(guidance),
            "--base_seed",
            str(seed),
            "--save_file",
            out_video,
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
        ]
        if negative_prompt:
            cmd.extend(["--negative_prompt", negative_prompt])
        log.info("  Generating: %s (seed %d, %d frames ~%.0fs)", Path(out_video).name, seed, frames, duration_s)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            err = (res.stderr or res.stdout or "")[:800]
            raise RuntimeError(f"WAN generation failed: {err}")
        if Path(out_video).exists():
            size_mb = Path(out_video).stat().st_size / 1024 / 1024
            log.info("    Generated %.1f MB (~%.0fs video)", size_mb, duration_s)
        else:
            raise RuntimeError("WAN command succeeded but output video missing")
    log.info("[STAGE 2/3] WAN 2.2 UNLOADED")


def validate_videos(task: str, prompt: str, videos: list[str], json_paths: list[str], reason2_python: Path):
    if not reason2_python.exists():
        raise RuntimeError(f"Reason2 python missing: {reason2_python}")
    if not VALIDATOR.exists():
        raise RuntimeError(f"Validator missing: {VALIDATOR}")

    log.info("[STAGE 3/3] Loading Cosmos-Reason2...")
    for video_path, out_json in zip(videos, json_paths):
        Path(out_json).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            str(reason2_python),
            str(VALIDATOR),
            "--video",
            str(Path(video_path).resolve()),
            "--task",
            task,
            "--prompt",
            prompt,
            "--output",
            out_json,
            "--fps",
            "4",
        ]
        log.info("  Validating: %s", Path(video_path).name)
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            warn = (res.stderr or res.stdout or "")[:300]
            log.warning("    Validation warning: %s", warn)
        else:
            log.info("    Validation complete")
    log.info("[STAGE 3/3] Cosmos-Reason2 UNLOADED")


def resolve_reason2_python(reason2_python: Path) -> Path:
    if reason2_python.exists():
        return reason2_python
    # Common typo fallback: ".../bin/pytho" -> ".../bin/python"
    if reason2_python.name == "pytho":
        candidate = reason2_python.with_name("python")
        if candidate.exists():
            log.warning("Reason2 python path typo detected, using %s", candidate)
            return candidate
    return reason2_python


def _sample_frame_count(sample_index: int, base_frames: int, alternate: bool) -> int:
    """Return frame count for a sample.

    When alternate=True:
        Even-numbered samples (2, 4, ...) → 5 seconds  = 121 frames
        Odd-numbered samples  (1, 3, 5, ...) → 10 seconds = 242 frames
    When alternate=False:
        All samples use base_frames.
    """
    if not alternate:
        return base_frames
    sample_num = sample_index + 1  # 1-indexed
    return 242 if sample_num % 2 == 1 else 121  # odd=10s (242f), even=5s (121f)


def process_single_image(
    image_path: str,
    task: str,
    num_samples: int,
    seed_start: int,
    qwen_env: str,
    wan_env: str,
    reason2_python: Path,
    frames: int,
    steps: int,
    guidance: float,
    size: str,
    no_validate: bool,
    qwen_system_prompt: Optional[str],
    alternate_duration: bool = False,
):
    image = Path(image_path)
    if not image.exists():
        log.error("Image not found: %s", image)
        return None

    prompt = filename_to_prompt(image)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_BASE / f"nav_{task}_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    log.info("\n%s", "=" * 70)
    log.info("Image: %s", image.name)
    log.info("Task: %s", task)
    log.info("Prompt: \"%s\"", prompt)
    log.info("Samples: %d%s", num_samples,
             " (alternating: odd=10s/242f, even=5s/121f)" if alternate_duration else "")
    log.info("%s\n", "=" * 70)

    outputs = []  # list of (video_path, seed, frame_count)
    enhanced = prompt
    negative_prompt = None
    try:
        config_data = {
            "run_id": run_id,
            "task": task,
            "image": str(image),
            "num_samples": num_samples,
            "seed_start": seed_start,
            "qwen_conda_env": qwen_env,
            "qwen_system_prompt": qwen_system_prompt,
            "wan_conda_env": wan_env,
            "reason2_python": str(reason2_python),
            "frames": frames,
            "alternate_duration": alternate_duration,
            "steps": steps,
            "guidance": guidance,
            "size": size,
            "no_validate": no_validate,
        }
        write_simple_yaml(run_dir / "config.yaml", config_data)

        if task in ("ground", "g1_nav") and qwen_system_prompt:
            log.info("Using custom Qwen system prompt for %s navigation.", task)
        enhanced, negative_prompt = enhance_prompt(task, prompt, str(image), qwen_env, qwen_system_prompt=qwen_system_prompt)

        for i in range(num_samples):
            seed = seed_start + i
            sample_frames = _sample_frame_count(i, frames, alternate_duration)
            duration_s = sample_frames / 24
            duration_tag = f"{duration_s:.0f}s"
            out_video = run_dir / f"nav_{task}_{run_id}_s{seed}_{duration_tag}.mp4"
            outputs.append((str(out_video), seed, sample_frames))

        generate_videos(
            image_path=str(image),
            prompt=enhanced,
            negative_prompt=negative_prompt,
            outputs=outputs,
            wan_env=wan_env,
            steps=steps,
            guidance=guidance,
            size=size,
        )

        # Write sample sidecars immediately after generation so they exist
        # even if validation later fails.
        for out_video, seed, sample_frames in outputs:
            sample_name = Path(out_video).stem
            sample_json = run_dir / f"{sample_name}.json"
            write_sample_json(
                sample_json_path=sample_json,
                name=sample_name,
                prompt=enhanced,
                seed=seed,
                guidance=guidance,
                image_path=str(image),
                output_video=out_video,
                negative_prompt=negative_prompt,
                num_frames=sample_frames,
                steps=steps,
                size=size,
                validation_path=None,
            )

        validation_map = {}
        if not no_validate:
            videos = [v for v, _, _ in outputs]
            validation_jsons = [str(run_dir / f"validation_{Path(v).stem}.json") for v in videos]
            validate_videos(task, prompt, videos, validation_jsons, reason2_python=reason2_python)
            validation_map = {v: j for v, j in zip(videos, validation_jsons)}

        for out_video, seed, sample_frames in outputs:
            sample_name = Path(out_video).stem
            sample_json = run_dir / f"{sample_name}.json"
            write_sample_json(
                sample_json_path=sample_json,
                name=sample_name,
                prompt=enhanced,
                seed=seed,
                guidance=guidance,
                image_path=str(image),
                output_video=out_video,
                negative_prompt=negative_prompt,
                num_frames=sample_frames,
                steps=steps,
                size=size,
                validation_path=validation_map.get(out_video),
            )

        result = {
            "run_id": run_id,
            "image": str(image),
            "task": task,
            "prompt": prompt,
            "enhanced_prompt": enhanced,
            "negative_prompt": negative_prompt,
            "samples": num_samples,
            "alternate_duration": alternate_duration,
            "seed_start": seed_start,
            "qwen_env": qwen_env,
            "qwen_system_prompt": qwen_system_prompt,
            "wan_env": wan_env,
            "reason2_python": str(reason2_python),
            "validated": not no_validate,
            "outputs": [
                {"video": v, "seed": s, "frames": f, "duration_s": f / 24}
                for v, s, f in outputs
            ],
        }
        with open(run_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        log.info("\n✓ Image complete. Results: %s\n", run_dir)
        return result
    except Exception as e:
        log.error("✗ Image failed: %s", e)
        for out_video, seed, sample_frames in outputs:
            sample_json = run_dir / f"{Path(out_video).stem}.json"
            if not sample_json.exists():
                write_sample_json(
                    sample_json_path=sample_json,
                    name=Path(out_video).stem,
                    prompt=enhanced,
                    seed=seed,
                    guidance=guidance,
                    image_path=str(image),
                    output_video=out_video,
                    negative_prompt=negative_prompt,
                    num_frames=sample_frames,
                    steps=steps,
                    size=size,
                    validation_path=None,
                )
        fail_report = {
            "run_id": run_id,
            "image": str(image),
            "task": task,
            "prompt": prompt,
            "samples": num_samples,
            "alternate_duration": alternate_duration,
            "seed_start": seed_start,
            "qwen_env": qwen_env,
            "qwen_system_prompt": qwen_system_prompt,
            "wan_env": wan_env,
            "reason2_python": str(reason2_python),
            "validated": not no_validate,
            "outputs": [v for v, _, _ in outputs],
            "error": str(e),
        }
        with open(run_dir / "report.json", "w", encoding="utf-8") as f:
            json.dump(fail_report, f, indent=2)
        return None


def process_all_images(
    input_dir: str,
    task: str,
    num_samples: int,
    seed_start: int,
    qwen_env: str,
    wan_env: str,
    reason2_python: Path,
    frames: int,
    steps: int,
    guidance: float,
    size: str,
    max_images: int | None,
    no_validate: bool,
    qwen_system_prompt: Optional[str],
    alternate_duration: bool = False,
):
    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        images.extend(sorted(glob.glob(str(Path(input_dir) / ext))))
    images = sorted(images)
    if max_images is not None:
        images = images[:max_images]

    total_videos = len(images) * num_samples
    log.info("Found %d images to process in %s", len(images), input_dir)
    log.info("Total videos to generate: %d (%d per image)", total_videos, num_samples)
    if alternate_duration:
        odd_count = (num_samples + 1) // 2
        even_count = num_samples // 2
        log.info("Duration pattern: %d x 10s (odd) + %d x 5s (even) per image", odd_count, even_count)
        est_per_image = odd_count * 15 + even_count * 6 + 5  # rough minutes (gen + enhance + validate)
        log.info("Estimated time: ~%d min/image, ~%.1f hours total",
                 est_per_image, (est_per_image * len(images)) / 60)

    results = []
    for idx, image_path in enumerate(images, 1):
        log.info("\n[%d/%d] Processing...", idx, len(images))
        r = process_single_image(
            image_path=image_path,
            task=task,
            num_samples=num_samples,
            seed_start=seed_start,
            qwen_env=qwen_env,
            wan_env=wan_env,
            reason2_python=reason2_python,
            frames=frames,
            steps=steps,
            guidance=guidance,
            size=size,
            no_validate=no_validate,
            qwen_system_prompt=qwen_system_prompt,
            alternate_duration=alternate_duration,
        )
        if r:
            results.append(r)

    log.info("\n%s", "=" * 70)
    log.info("BATCH COMPLETE: %d/%d images processed", len(results), len(images))
    log.info("Total videos generated: ~%d", len(results) * num_samples)
    log.info("%s\n", "=" * 70)
    return results


def main():
    p = argparse.ArgumentParser(description="GPU-optimized sequential navigation batch runner")
    p.add_argument("--image", type=str, default=None, help="Single image (or use --all)")
    p.add_argument("--all", action="store_true", help=f"Process all images in {DEFAULT_NAV_DIR}")
    p.add_argument("--input-dir", type=str, default=DEFAULT_NAV_DIR, help="Navigation image directory")
    p.add_argument("--max-images", type=int, default=None, help="Limit number of images when using --all")
    p.add_argument("--task", type=str, choices=["drone", "ground", "g1_nav"], default="ground")
    p.add_argument("--num-samples", type=int, default=5, help="Videos per image (default: 5)")
    p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--qwen-conda-env", type=str, default=DEFAULT_QWEN_ENV)
    p.add_argument("--wan-conda-env", type=str, default=DEFAULT_WAN_ENV)
    p.add_argument("--reason2-python", type=str, default=str(REASON2_VENV_PY))
    p.add_argument("--frames", type=int, default=121,
                   help="Base frame count (default: 121 = ~5s). Overridden per-sample when --alternate-duration is on.")
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--guidance", type=float, default=7.5)
    p.add_argument("--size", type=str, default="1280*704")
    p.add_argument("--no-validate", action="store_true", help="Skip validation stage")
    p.add_argument("--alternate-duration", action="store_true", default=True,
                   help="Alternate video duration: odd samples=10s (242 frames), "
                        "even samples=5s (121 frames). Enabled by default.")
    p.add_argument("--no-alternate-duration", action="store_true",
                   help="Disable alternating duration — all samples use --frames.")
    p.add_argument(
        "--qwen-system-prompt",
        type=str,
        default=None,
        help="Override Qwen system prompt. If omitted, each extender uses its own research-optimized prompt.",
    )
    args = p.parse_args()

    reason2_python = resolve_reason2_python(Path(args.reason2_python))
    # Let extenders use their own research-optimized system prompts by default.
    # Only override if the user explicitly passes --qwen-system-prompt.
    effective_system_prompt = args.qwen_system_prompt

    # --no-alternate-duration disables the default alternation
    alternate_duration = args.alternate_duration and not args.no_alternate_duration

    if args.all:
        process_all_images(
            input_dir=args.input_dir,
            task=args.task,
            num_samples=args.num_samples,
            seed_start=args.seed_start,
            qwen_env=args.qwen_conda_env,
            wan_env=args.wan_conda_env,
            reason2_python=reason2_python,
            frames=args.frames,
            steps=args.steps,
            guidance=args.guidance,
            size=args.size,
            max_images=args.max_images,
            no_validate=args.no_validate,
            qwen_system_prompt=effective_system_prompt,
            alternate_duration=alternate_duration,
        )
    elif args.image:
        process_single_image(
            image_path=args.image,
            task=args.task,
            num_samples=args.num_samples,
            seed_start=args.seed_start,
            qwen_env=args.qwen_conda_env,
            wan_env=args.wan_conda_env,
            reason2_python=reason2_python,
            frames=args.frames,
            steps=args.steps,
            guidance=args.guidance,
            size=args.size,
            no_validate=args.no_validate,
            qwen_system_prompt=effective_system_prompt,
            alternate_duration=alternate_duration,
        )
    else:
        p.print_help()


if __name__ == "__main__":
    main()
