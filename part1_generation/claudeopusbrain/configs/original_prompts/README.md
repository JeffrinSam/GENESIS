# Original System Prompts — Backup

Saved on: 2026-02-08

These are the **original unmodified** system prompts before any self-tuning iterations.
They serve as:
1. Baseline for comparison in experiments
2. Fallback if self-tuning degrades quality
3. Reference for the IROS paper (Manual baseline)

## Files

| File | Source | Purpose |
|------|--------|---------|
| `g1_system_prompt.txt` | `Qwen3-VL/prompt_extenders/cosmos25/prompt_extender_unitree_g1.py` | G1_SYSTEM_PROMPT — Qwen3-VL guidance for humanoid manipulation |
| `g1_negative_prompt.txt` | same file | G1_NEGATIVE_PROMPT — Cosmos diffusion model negative prompt |
| `claude_optimizer_system_prompt.txt` | `Claudeopusbrain/src/claude_brain.py` | Claude Opus 4.6 optimizer role definition |
| `llama_optimizer_system_prompt.txt` | `Claudeopusbrain/src/opensource_optimizer.py` | Llama 3.1 optimizer role definition |

## Source Paths

- G1 extender: `/mnt/Thesis/JeffrinSam/Part1/Qwen3-VL/prompt_extenders/cosmos25/prompt_extender_unitree_g1.py`
- Claude brain: `/mnt/Thesis/JeffrinSam/Part1/Claudeopusbrain/src/claude_brain.py` (line 198-261)
- Llama optimizer: `/mnt/Thesis/JeffrinSam/Part1/Claudeopusbrain/src/opensource_optimizer.py` (line 188-224)
