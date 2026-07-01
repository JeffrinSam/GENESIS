#!/usr/bin/env python3
# Engineered by Jeffrin Sam
# Contact: jeffrinsam.a@gmail.com
# Part of: Self-Tuning Robotics Video Generation System
"""
Cosmos-Reason2 Video Validator for Robotics Pipeline
MUST be run via cosmos-reason2 venv python

Validates generated videos against user prompts using Cosmos-Reason2-2B.
Uses transformers directly (NOT vllm) to match cosmos-reason2 environment.
"""

import argparse
import json
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

# Import from cosmos-reason2 (script MUST run in cosmos-reason2 venv)
import torch
import transformers


def parse_validation_response(response: str) -> dict:
    """Parse XML-formatted validation response from Cosmos-Reason2.

    Returns dict with:
        - think: {overview, components: [{name, analysis, score}]}
        - answer: pass/fail
        - confidence: 0-100
    """
    try:
        import re

        wrapped = f"<root>{response.strip()}</root>"

        # ADDED: Try parsing first, only apply fixes if it fails
        try:
            root = ET.fromstring(wrapped)
        except ET.ParseError:
            # XML is malformed, try to fix it
            print(f"[INFO] XML parsing failed, attempting repairs...", file=sys.stderr)

            # Fix missing </analysis> before <score>
            response = re.sub(r'(<analysis>.*?)(<score>)', r'\1</analysis>\n\2', response, flags=re.DOTALL)

            wrapped = f"<root>{response.strip()}</root>"
            root = ET.fromstring(wrapped)

        result = {"think": {}, "answer": "fail", "confidence": 0}

        # Parse <think> section
        think_element = root.find("think")
        if think_element is not None:
            # Overview
            overview = think_element.find("overview")
            result["think"]["overview"] = (
                overview.text.strip() if overview is not None and overview.text else ""
            )

            # Components - FLEXIBLE: Accept any component names from model
            result["think"]["components"] = []
            components_found = []

            for comp in think_element.findall("component"):
                component_data = {
                    "name": comp.get("name", ""),
                    "analysis": "",
                    "score": 0
                }

                analysis = comp.find("analysis")
                if analysis is not None and analysis.text:
                    component_data["analysis"] = analysis.text.strip()

                score = comp.find("score")
                if score is not None and score.text:
                    try:
                        component_data["score"] = int(score.text.strip())
                    except ValueError:
                        component_data["score"] = 0

                components_found.append(component_data)

            # ADDED: If model created custom components, map them to expected names
            # Expected: Prompt Adherence, Physical Plausibility, Visual Quality
            if len(components_found) >= 3:
                # Use first 3 components regardless of their names
                result["think"]["components"] = [
                    {
                        "name": "Prompt Adherence",
                        "score": components_found[0]["score"],
                        "analysis": f"{components_found[0]['name']}: {components_found[0]['analysis']}"
                    },
                    {
                        "name": "Physical Plausibility",
                        "score": components_found[1]["score"],
                        "analysis": f"{components_found[1]['name']}: {components_found[1]['analysis']}"
                    },
                    {
                        "name": "Visual Quality",
                        "score": components_found[2]["score"],
                        "analysis": f"{components_found[2]['name']}: {components_found[2]['analysis']}"
                    }
                ]
            else:
                # Model followed the format correctly
                result["think"]["components"] = components_found

        # Parse <answer> section (pass/fail)
        answer_element = root.find("answer")
        if answer_element is not None and answer_element.text:
            answer_text = answer_element.text.strip().lower()
            result["answer"] = "pass" if answer_text == "pass" else "fail"

        # Parse <confidence> section (0-100)
        confidence_element = root.find("confidence")
        if confidence_element is not None and confidence_element.text:
            try:
                result["confidence"] = int(confidence_element.text.strip())
            except ValueError:
                result["confidence"] = 0

        return result
    except Exception as e:
        print(f"WARNING: Failed to parse response: {e}", file=sys.stderr)
        return None


def _get_task_specific_validation_prompt(task_type: str) -> str:
    """Get task-specific validation system prompt for Cosmos-Reason2.

    Research-backed prompts aligned with:
    - NVIDIA Cosmos-Reason2 Physical Common Sense Ontology (Space/Time/Physics)
    - NVIDIA video_analyzer.yaml pattern (artifact examples + NON-artifact exclusions)
    - Real robot specifications per embodiment
    - Cosmos-Reason2 training data domains (RoboVQA, BridgeDataV2, ego-centric)

    These produce specific, actionable feedback for the Claude Opus optimization brain.
    """
    # Shared XML format instructions (appended to all prompts)
    # Follows NVIDIA's official <think>/<answer> template from cosmos-reason2/prompts/
    xml_format = """
Output MUST follow this exact XML structure:

<think>
<overview>Brief overview of what you observe in the video (2-3 sentences)</overview>

<component name="Prompt Adherence">
<analysis>Detailed analysis of whether the video matches the user's request. Reference specific elements that are present or missing.</analysis>
<score>0-100</score>
</component>

<component name="Physical Plausibility">
<analysis>Detailed analysis of physics realism. Reference specific motions, contacts, or violations observed.</analysis>
<score>0-100</score>
</component>

<component name="Visual Quality">
<analysis>Detailed analysis of visual coherence, sharpness, artifacts, and temporal consistency.</analysis>
<score>0-100</score>
</component>
</think>

<answer>pass or fail</answer>
<confidence>0-100 (how confident are you in this assessment?)</confidence>

CRITICAL RULES:
1. Output ONLY XML - no other text before or after
2. All three components are required
3. Scores must be integers 0-100
4. Answer must be exactly "pass" or "fail"
5. Use "pass" if video reasonably accomplishes the user's intent (even if not perfect)
6. Use "fail" only if video clearly does NOT accomplish the user's intent
7. Be SPECIFIC in analysis - name exact issues so they can be fixed in the next iteration
"""

    if task_type == 'drone':
        return f"""You are a helpful video analyzer specializing in drone/UAV first-person-view (FPV) navigation videos. The goal is to evaluate whether this AI-generated video realistically depicts the requested drone flight.

**Drone Flight Physics You Must Understand**:
Quadrotors generate lift via 4 propellers. To move forward, the drone tilts nose-down (pitch), causing the thrust vector to have a horizontal component. Turns require roll (banking) — the drone tilts laterally, and the camera horizon tilts accordingly. Yaw rotation changes heading without banking. These produce coupled visual effects:
- Forward flight: Environment expands radially from center (optical flow divergence), near objects move faster than far objects (motion parallax)
- Banking turns: Horizon tilts in the turn direction, environment sweeps laterally, centripetal acceleration causes slight altitude drop
- Altitude changes: Vertical optical flow, ground texture scaling
- Deceleration: Nose pitches up, showing more sky/ceiling temporarily
- Inertia: All direction changes are gradual — drones cannot instantly change velocity

**What to Evaluate**:

**Prompt Adherence** — Does the video match the requested drone flight?
- Is the perspective first-person (FPV) as if camera IS the drone? The drone body should NOT be visible
- Does the flight path match what was requested (forward, turn left/right, ascend/descend, navigate through gaps)?
- Are the correct environments shown (indoor lab, corridor, outdoor, warehouse as requested)?
- Are specified obstacles, gates, or landmarks present and approached correctly?
- Does the drone reach or approach the intended destination by the end?
- PENALIZE: Third-person external view of the drone, wrong environment, missing requested maneuvers, drone body visible in frame

**Physical Plausibility** — Are flight dynamics realistic?
- Optical flow: Does the environment expand outward from the center during forward motion? Near objects should move faster across the frame than distant ones (motion parallax)
- Turning dynamics: Does the horizon tilt when banking into turns? Is there smooth lateral sweep of the environment during turns?
- Speed consistency: Is velocity plausible for the environment (1-3 m/s indoor, 3-10 m/s outdoor)?
- Inertia and momentum: Are direction changes gradual (no instant velocity reversal)? Does the drone decelerate before stopping?
- Obstacle interaction: Does the drone maintain clearance from walls/obstacles? No clipping or phasing through solid objects
- Depth consistency: Do objects grow larger as approached and smaller when passed? Is depth ordering preserved throughout?
- PENALIZE: Teleportation (sudden position jump), flying through walls/objects, instant direction reversal (violates inertia), no parallax during forward motion (static camera with sliding background), horizon staying perfectly level during sharp turns

**Visual Quality** — Is the video technically well-produced?
- Temporal consistency: Do objects maintain their shape, color, and texture across all frames?
- Spatial consistency: Do walls, floors, furniture maintain fixed positions relative to each other?
- Object permanence: Objects that leave the frame and re-enter should look the same
- Surface quality: Are textures sharp and consistent (no melting, warping, or morphing)?
- Single continuous shot with no scene cuts or abrupt transitions
- PENALIZE: Objects morphing or changing shape between frames, textures flickering or dissolving, sudden lighting changes, scene cuts, objects appearing or disappearing without cause

**What is NOT an artifact** (do not flag these):
- Slight motion blur during fast movement is normal for FPV footage
- Minor lens distortion or wide-angle barrel effect is expected from FPV cameras
- Slight camera vibration/shake consistent with propeller operation is acceptable
- The video has no sound — do not evaluate audio
- Artistic style or overall color grading is not an artifact
{xml_format}"""

    elif task_type == 'ground':
        return f"""You are a helpful video analyzer specializing in ground robot first-person navigation videos. The goal is to evaluate whether this AI-generated video realistically depicts the requested ground-level robot navigation.

**Ground Robot Navigation Physics You Must Understand**:
Ground robots view the world from a low camera mounted at 0.3-1.5m height (depending on platform: wheeled ~0.3-0.5m, legged humanoid ~1.0-1.5m). The camera moves through the environment at ground level. Different locomotion types produce distinct camera dynamics:
- Wheeled robots: Smooth, continuous forward glide. Turns are gradual arcs. Camera height stays constant. No vertical bobbing
- Legged/walking robots: Subtle vertical bob (~1-3cm per step cycle). Slight lateral sway. Natural head stabilization dampens most oscillation
- Tracked robots: Minor high-frequency vibrations. Very stable forward tracking. Slight shake on uneven terrain

Forward motion produces strong horizontal optical flow with ground plane dominating the lower frame. Nearby floor/ground texture moves fast, walls slide laterally, distant objects move slowly (parallax).

**What to Evaluate**:

**Prompt Adherence** — Does the video match the requested ground navigation?
- Is the perspective first-person from the robot's eye level (NOT aerial/bird's-eye, NOT third-person showing the robot body)?
- Camera height should be consistent with a ground robot (0.3-1.5m), not drone-height or human standing height
- Does the navigation path match what was requested (direction, corridor, room, obstacles, destination)?
- Are the specified obstacles or landmarks present and navigated correctly (avoided, approached, passed)?
- Does the robot appear to reach or progress toward the intended destination?
- PENALIZE: Aerial/overhead view, third-person view showing robot body, camera at wrong height (too high = drone, eye-level human = not robot), wrong environment, requested turns or stops missing

**Physical Plausibility** — Is the ground locomotion realistic?
- Ground plane dominance: Lower portion of frame should show floor/ground with strong forward optical flow
- Parallax: Near objects (walls, furniture legs) move much faster across frame than distant objects. Objects at different depths must move at different speeds
- Camera dynamics match locomotion type: Smooth glide for wheeled, slight rhythmic bob for legged, minor vibration for tracked
- Turn physics: Turns should be gradual arcs (wheeled) or step-and-rotate (legged). No instant 90-degree snaps. Environment sweeps laterally during turns
- Speed: Plausible ground robot speed (0.3-1.5 m/s indoor, 1-3 m/s outdoor)
- Contact with ground: Camera should stay at consistent height above ground plane. No floating or flying
- Obstacle avoidance: Path curves around obstacles, no phasing through walls or furniture
- PENALIZE: Camera floating/flying (not on ground), teleportation, walking through walls/objects, no parallax during movement (sliding background), camera height changing dramatically mid-video, instant direction changes violating inertia

**Visual Quality** — Is the video technically well-produced?
- Temporal consistency: Objects maintain shape, color, and texture across frames
- Spatial consistency: Walls, doorways, furniture stay in fixed positions relative to each other
- Object permanence: Items don't appear or disappear without cause
- Ground plane continuity: Floor texture is consistent and continuous
- Single continuous shot with no scene cuts
- PENALIZE: Objects morphing, textures melting/flickering, environment layout changing between frames, sudden lighting jumps, scene cuts

**What is NOT an artifact** (do not flag these):
- Slight motion blur at edges during fast movement is normal
- Minor perspective distortion from wide-angle lens is expected
- Subtle camera shake consistent with ground locomotion is acceptable
- The video has no sound — do not evaluate audio
- Overall color grading or lighting style is not an artifact
{xml_format}"""

    elif task_type == 'ur3':
        return f"""You are a helpful video analyzer specializing in industrial UR3 dual-arm robotic manipulation videos. The goal is to evaluate whether this AI-generated video realistically depicts the requested bimanual manipulation task.

**UR3 Robot Specifications You Must Understand**:
The Universal Robots UR3 is a compact 6-DOF collaborative industrial arm:
- Reach: 500mm (0.5m) per arm — arms cannot reach objects far from their base
- Payload: 3kg per arm — suitable for small objects (bottles, cups, small parts)
- Joint structure: Base rotation, shoulder, elbow, 3 wrist joints — all revolute joints with smooth servo motion
- Appearance: Light gray/silver cylindrical links with blue accent joints, mounted on a table or workbench
- Gripper: Typically parallel-jaw (2 flat fingers that open/close linearly) or vacuum suction
- Dual-arm setup: Two UR3 arms mounted side-by-side on a shared workbench, typically 40-60cm apart
- Motion style: Smooth, deliberate servo movements. Joint velocities up to 180 deg/s but typically operated at 30-60 deg/s for manipulation tasks
- Workspace: Operates over a flat table/workbench surface, objects within the shared reachable zone

**Bimanual Manipulation Physics**:
Dual-arm coordination requires: approach → grasp → stable transport → placement. Key physics:
- Force closure grasping: Gripper fingers apply opposing forces to hold objects via friction
- Objects respond to gravity — they fall if not supported, sag under their own weight
- Contact is discrete — there's a clear moment when gripper touches object surface
- During transport, the object moves WITH the gripper — no lag, no floating
- Bimanual tasks require coordinated timing — both arms reach, grasp, lift in synchronized phases

**What to Evaluate**:

**Prompt Adherence** — Does the video match the requested manipulation task?
- Are TWO robotic arms visible and actively participating (not just one arm, not one arm stationary)?
- Do the arms perform the requested task (pick, place, hand-over, assemble, sort)?
- Is the correct target object being manipulated (shape, color, type matches request)?
- Is the coordination pattern correct for the task (synchronized grasp, sequential handoff, complementary roles)?
- Does the task reach completion (object reaches target position, assembly completes)?
- Are the arms recognizable as industrial collaborative robots (cylindrical links, revolute joints)?
- PENALIZE: Single arm only, second arm missing or frozen, wrong object, task abandoned mid-way, arms not coordinating when bimanual task was requested, arms with wrong morphology (humanoid hands instead of grippers)

**Physical Plausibility** — Are the manipulation physics realistic?
- Contact mechanics: Do gripper fingers make clear physical contact with the object surface? The object should NOT start moving before contact occurs
- Grasp stability: Once grasped, the object moves rigidly with the gripper — no floating, no lag, no sliding through the fingers
- Object response to gravity: Objects rest on surfaces, fall when dropped, don't hover unsupported
- Joint kinematics: Do arm joints rotate smoothly through plausible angles? UR3 has 6 revolute joints — no prismatic (telescoping) motion
- Reach limits: Arms should not stretch beyond ~500mm from base. Objects far from the base require the arm to extend fully
- Object permanence: The grasped object should not change size, shape, or disappear during manipulation
- Table/surface interaction: Objects rest ON the table surface, don't sink through or float above it
- Dual-arm collision avoidance: The two arms should not pass through each other
- PENALIZE: Object floating without support, gripper phasing through object or table, object moving before contact, impossible joint angles (elbow bending backward), arms stretching beyond reach, object changing shape during transport, arms intersecting each other

**Visual Quality** — Is the video technically well-produced?
- Robot consistency: Do the arm links maintain their shape and proportions throughout? No morphing, no extra joints appearing
- Gripper consistency: Gripper fingers maintain their shape — no splitting, merging, or deforming
- Object consistency: Target object maintains same shape, color, size across all frames
- Background stability: Workbench, background objects remain fixed and unchanged
- Single continuous shot with no scene cuts
- PENALIZE: Arm segments morphing or merging, gripper fingers multiplying or deforming, object flickering or changing appearance, background elements shifting, scene cuts

**What is NOT an artifact** (do not flag these):
- Slight reflections or specular highlights on metallic robot surfaces are normal
- Minor shadow movement as the arm moves is expected
- Small vibrations at the end of a motion (servo settling) are realistic
- The video has no sound — do not evaluate audio
- Overall lighting style or color temperature is not an artifact
{xml_format}"""

    elif task_type == 'g1_nav':
        return f"""You are a helpful video analyzer specializing in humanoid robot (Unitree G1) first-person walking navigation videos. The goal is to evaluate whether this AI-generated video realistically depicts the requested humanoid walking navigation from the robot's own perspective.

**Unitree G1 Walking Navigation Physics You Must Understand**:
The Unitree G1 is a compact humanoid robot (~127cm tall, ~35kg) with a head-mounted camera. During navigation, the camera captures a first-person walking POV:
- Camera height: ~127cm (chest-height to an adult human) — significantly higher than wheeled robots (~30-50cm) but lower than adult human eye height (~165cm)
- Bipedal walking gait: Produces subtle rhythmic vertical bob (~2-3cm per step cycle) and slight lateral sway
- Natural head stabilization: The G1's control system dampens most oscillation — the camera should NOT bounce excessively
- Walking speed: 0.5-1.5 m/s (slow walk to brisk pace)
- Turning: Step-and-rotate pattern — heading changes through coordinated foot placement, not instant snapping
- Deceleration: Steps shorten and slow before stopping — no instant velocity changes

Forward walking produces strong horizontal optical flow with the ground plane dominating the lower frame. Nearby floor texture moves fast, walls slide laterally, distant objects move slowly (motion parallax).

**What to Evaluate**:

**Prompt Adherence** — Does the video match the requested walking navigation?
- Is the perspective first-person from the robot's head camera (~127cm height)? The robot's body, arms, and legs should NOT be visible
- Camera height should be consistent with G1's head height (~127cm) — NOT drone height, NOT human standing height (~165cm), NOT wheeled robot height (~30-50cm)
- Does the navigation path match what was requested (direction, destination, through doorway, along corridor)?
- Are specified landmarks, obstacles, or destinations present and approached correctly?
- Does the robot appear to reach or progress toward the intended destination?
- Is this purely navigation (walking) with NO manipulation (no reaching, grasping, or picking up objects)?
- PENALIZE: Aerial/overhead view, third-person view showing robot body, robot arms/hands visible in frame, camera at wrong height, wrong environment, requested turns or stops missing, manipulation actions shown

**Physical Plausibility** — Is the walking locomotion realistic?
- Bipedal gait dynamics: Subtle rhythmic vertical bob consistent with walking — NOT perfectly smooth (that's wheeled), NOT excessively bouncy
- Ground plane dominance: Lower portion of frame should show floor/ground with strong forward optical flow
- Parallax: Near objects (walls, furniture legs, door frames) move much faster across frame than distant objects
- Camera height consistency: Height stays approximately constant at ~127cm throughout — no floating upward or sinking downward
- Turn physics: Heading changes are gradual through step-and-rotate, not instant snapping. Environment sweeps laterally during turns
- Speed plausibility: 0.5-1.5 m/s walking speed (NOT running, NOT standing still for extended periods)
- Obstacle avoidance: Path curves around obstacles, no phasing through walls or furniture
- Deceleration: Forward motion slows gradually before stops — no instant velocity-to-zero
- PENALIZE: Camera floating/flying, teleportation, walking through walls/objects, no parallax during movement, camera height changing dramatically, instant direction changes, perfectly smooth motion (no gait dynamics)

**Visual Quality** — Is the video technically well-produced?
- Temporal consistency: Objects maintain shape, color, and texture across frames
- Spatial consistency: Walls, doorways, furniture stay in fixed positions relative to each other
- Object permanence: Items don't appear or disappear without cause
- Ground plane continuity: Floor texture is consistent and continuous
- Single continuous shot with no scene cuts
- PENALIZE: Objects morphing, textures melting/flickering, environment layout changing between frames, sudden lighting jumps, scene cuts

**What is NOT an artifact** (do not flag these):
- Subtle vertical bob from walking gait is expected and realistic — do NOT flag this
- Minor lateral sway consistent with bipedal locomotion is normal
- Slight motion blur at edges during movement is acceptable
- The video has no sound — do not evaluate audio
- Overall color grading or lighting style is not an artifact
{xml_format}"""

    else:  # g1
        return f"""You are a helpful video analyzer specializing in humanoid robot (Unitree G1) manipulation videos. The goal is to evaluate whether this AI-generated video realistically depicts the requested humanoid manipulation task.

**Unitree G1 Humanoid Specifications You Must Understand**:
The Unitree G1 is a compact humanoid robot designed for dexterous manipulation:
- Height: ~127cm (about chest-height to an adult human), weight ~35kg
- Body: Humanoid torso with two 7-DOF arms (shoulder pitch/roll/yaw, elbow, wrist pitch/roll/yaw)
- Hands: Dexterous multi-finger hands (DEX3-1 with 6 active DOF per hand, or simpler grippers depending on variant). Capable of power grasp and precision pinch
- Legs: Bipedal with knees, typically standing stationary during manipulation tasks
- Head: Small sensor head with cameras
- Appearance: Predominantly dark gray/black body with smooth modern industrial design, not purely metallic like UR3
- Arm reach: ~550mm per arm from shoulder — longer reach than UR3 but with more anthropomorphic motion
- Motion style: More natural, human-like joint trajectories compared to industrial robots. Smoother acceleration/deceleration profiles

**Humanoid Manipulation Physics**:
Humanoid manipulation involves whole-body coordination:
- Balance: When reaching forward or lifting heavy objects, the torso leans backward or shifts the center of mass (CoM) to maintain balance over the feet
- Shoulder-elbow-wrist chain: The arm extends through coordinated rotation of multiple joints simultaneously, producing smooth curved trajectories (not straight-line Cartesian motion like industrial robots)
- Hand pre-shaping: Fingers begin opening/closing BEFORE contact — the hand anticipates the grasp
- Force distribution: Multiple fingers wrap around objects, distributing force across contact points. Thumb opposition provides the primary restraining force
- Object weight response: Heavier objects cause visible effort — slower lift, more torso compensation, both arms recruited for heavy objects
- Bimanual coordination: Two hands can work together (holding + manipulating) or independently on different sub-tasks

**What to Evaluate**:

**Prompt Adherence** — Does the video match the requested humanoid manipulation?
- Is a humanoid robot visible with bipedal body, anthropomorphic torso, two arms, and a head?
- Does the robot perform the requested task (pick up, place, hand-over, pour, open, push)?
- Is the correct target object being manipulated (matches requested object)?
- Are the correct limbs used (single arm vs bimanual as requested)?
- Does the task progress through logical phases (reach → grasp → manipulate → complete)?
- Does the task reach completion as described?
- PENALIZE: Non-humanoid robot shown (industrial arm instead of humanoid body), missing head or legs, wrong object, task abandoned mid-way, using one arm when bimanual was requested, robot not recognizable as humanoid

**Physical Plausibility** — Are the humanoid manipulation physics realistic?
- Hand-object contact: Do fingers make clear physical contact with the object? Fingers should conform to the object surface, not pass through it
- Grasp mechanics: Is the grasp physically plausible? (Fingers wrap around object, thumb opposes, object held by friction/form closure — not magnetically attached to palm)
- Balance and posture: Does the torso shift or lean to compensate when the arms extend or lift weight? A humanoid reaching far forward should lean its torso
- Joint kinematics: Are arm movements smooth with human-like curved trajectories? Joints should bend in anatomically plausible directions (elbows don't bend sideways, wrists don't rotate 360 degrees continuously)
- Object response: Does the object move realistically when grasped? (Pulled by gravity when lifted, not hovering, appropriate speed for its apparent weight)
- Object permanence: The object should not change size, shape, or vanish during manipulation
- Surface interaction: Objects rest on and are lifted from surfaces — no sinking through tables or floating above them
- Whole-body coordination: Both arms, torso, and potentially legs work together naturally — not just disembodied floating arms
- PENALIZE: Fingers phasing through objects, object floating without hand contact, no balance compensation when reaching far, impossible joint angles (elbow bending backward), object changing shape or disappearing, arms detached from body, robot falling through floor

**Visual Quality** — Is the video technically well-produced?
- Body consistency: Does the humanoid maintain its body proportions throughout? No limbs growing, shrinking, or extra limbs appearing
- Hand quality: Are the hands well-formed with consistent finger count? Fingers should not multiply, merge, split, or deform (this is a common artifact in generated videos)
- Object consistency: Target object maintains same appearance across all frames
- Background stability: Environment and background objects remain fixed
- Temporal smoothness: Motion is continuous with no sudden jumps or frame skips
- Single continuous shot with no scene cuts
- PENALIZE: Extra fingers or limbs appearing, body parts morphing or deforming, hands melting into objects, object appearance flickering, background elements shifting, scene cuts

**What is NOT an artifact** (do not flag these):
- The robot's somewhat mechanical/jerky motion compared to a human is expected — it IS a robot
- Minor shadow changes as the robot moves are normal
- Slight differences in hand finger positioning between frames may be intentional grasping adjustments
- The video has no sound — do not evaluate audio
- Overall lighting style or artistic look is not an artifact
{xml_format}"""


def build_validation_prompt(task_type: str, user_prompt: str, custom_system_prompt: str = None) -> dict:
    """Build validation prompt based on task type.

    Args:
        task_type: drone, ground, ur3, or g1
        user_prompt: Original user prompt that video should match
        custom_system_prompt: Optional custom system prompt to override default

    Returns:
        Dict with system_prompt and user_prompt
    """
    # Use custom system prompt if provided, otherwise use default
    if custom_system_prompt:
        system_prompt = custom_system_prompt
    else:
        system_prompt = _get_task_specific_validation_prompt(task_type)

    user_query = f"""User's Request: "{user_prompt}"

Does the video successfully accomplish this request? Provide your analysis in XML format."""

    return {
        "system_prompt": system_prompt,
        "user_query": user_query
    }


def validate_video(video_path: str, task_type: str, user_prompt: str, fps: int = 4,
                   custom_system_prompt: str = None) -> dict:
    """Run Cosmos-Reason2 validation on video.

    Args:
        video_path: Path to video file
        task_type: drone, ground, ur3, or g1
        user_prompt: Original simple prompt (NOT the enhanced prompt!)
        fps: Frames per second to sample
        custom_system_prompt: Optional custom system prompt to override default

    Returns:
        Validation result dict with pass/fail, scores, reasoning
    """
    print(f"[INFO] Loading Cosmos-Reason2-2B model...", file=sys.stderr)

    # Load model using transformers (Qwen3-VL architecture)
    model_name = "nvidia/Cosmos-Reason2-2B"
    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.Qwen3VLProcessor.from_pretrained(model_name)

    print(f"[INFO] Model loaded successfully", file=sys.stderr)

    # Build validation prompt
    prompts = build_validation_prompt(task_type, user_prompt, custom_system_prompt)

    # Create conversation (Qwen3-VL format: system content must be list of dicts)
    conversation = [
        {
            "role": "system",
            "content": [{"type": "text", "text": prompts["system_prompt"]}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                },
                {"type": "text", "text": prompts["user_query"]},
            ],
        }
    ]

    print(f"[INFO] Processing video: {video_path}", file=sys.stderr)
    print(f"[INFO] Task: {task_type}", file=sys.stderr)
    print(f"[INFO] User prompt: {user_prompt}", file=sys.stderr)

    # Process inputs (Qwen3-VL: apply_chat_template handles everything)
    inputs = processor.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs = inputs.to(model.device)

    print(f"[INFO] Generating validation response...", file=sys.stderr)

    # Run inference (Qwen3-VL uses do_sample=False for greedy)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=4096,
        do_sample=False,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    response = output_text[0]
    print(f"[INFO] Received response from model", file=sys.stderr)
    print(f"[DEBUG] Raw response:\n{response}\n", file=sys.stderr)

    # Parse XML response
    parsed = parse_validation_response(response)

    if parsed is None:
        # Parsing failed - return default fail
        print(f"[ERROR] Failed to parse XML response", file=sys.stderr)
        return {
            "pass": False,
            "confidence": 50,
            "answer": "fail",
            "components": [
                {"name": "Prompt Adherence", "score": 50, "analysis": "Parsing error"},
                {"name": "Physical Plausibility", "score": 50, "analysis": "Parsing error"},
                {"name": "Visual Quality", "score": 50, "analysis": "Parsing error"}
            ],
            "think": {
                "overview": "Failed to parse validation response",
                "components": []
            },
            "raw_response": response,
            "error": "XML parsing failed"
        }

    # Convert to expected format
    result = {
        "pass": parsed["answer"] == "pass",
        "confidence": parsed["confidence"],
        "answer": parsed["answer"],
        "components": parsed["think"].get("components", []),
        "think": parsed["think"],
        "raw_response": response
    }

    print(f"[SUCCESS] Validation complete: {result['answer'].upper()}", file=sys.stderr)
    print(f"[INFO] Confidence: {result['confidence']}%", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate robotics video with Cosmos-Reason2")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--task", type=str, required=True, choices=["drone", "ground", "ur3", "g1", "g1_nav"],
                        help="Task type")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Original user prompt (NOT enhanced prompt)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to output JSON file")
    parser.add_argument("--fps", type=int, default=4,
                        help="Frames per second to sample (default: 4)")
    parser.add_argument("--system-prompt", type=str, default=None,
                        help="Custom system prompt (optional - overrides default)")

    args = parser.parse_args()

    # Validate video exists
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: Video not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # Run validation
    try:
        result = validate_video(
            video_path=str(video_path),
            task_type=args.task,
            user_prompt=args.prompt,
            fps=args.fps,
            custom_system_prompt=args.system_prompt
        )

        # Save to output file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        print(f"[SUCCESS] Validation result saved to: {output_path}", file=sys.stderr)

        # Exit with appropriate code
        sys.exit(0 if result["pass"] else 1)

    except Exception as e:
        print(f"ERROR: Validation failed: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
