# Quick Reference — Self-Tuning Video Generation

## 🚀 Tomorrow: 100-Task Experiments

### Step 1: Prepare Tasks File

```bash
cd /mnt/Thesis/JeffrinSam/Part1/Claudeopusbrain

# Copy template
cp tasks_template.json my_100_tasks.json

# Edit with your 50 manipulation + 50 navigation tasks
# Format:
# [
#   {"task": "G1 picks up bottle", "task_type": "g1", "image": "./images/g1_1.jpg"},
#   {"task": "UR3 places block", "task_type": "ur3", "image": "./images/ur3_1.jpg"},
#   ...
# ]
```

### Step 2: Run Batch Experiments

```bash
# Activate conda
conda activate wan2.2

# Set API key
export ANTHROPIC_API_KEY="your-key-here"

# Run with Claude Opus (best quality)
conda run -n wan2.2 python3 run_batch_experiments.py \
  --tasks my_100_tasks.json \
  --model opus \
  --max-iterations 5 \
  --cost-budget 200 \
  --output-dir ./results/iros2026

# Expected:
# - Runtime: 40-50 hours (30 min/task × 100 tasks)
# - Cost: ~$200 ($2/task × 100)
# - Success rate: 85% (85/100 reach 80+ score)
```

### Step 3: If Crash, Resume

```bash
# Crash happened? No problem — checkpoint auto-saved!
conda run -n wan2.2 python3 run_batch_experiments.py \
  --tasks my_100_tasks.json \
  --model opus \
  --resume

# Resumes from last checkpoint
# Full history preserved
# No lost work!
```

---

## 📊 Monitor Progress

While running, the script shows:

```
📊 PROGRESS: 45/100 tasks
   Elapsed: 22:30:15
   ETA: 25:15:00
   Total cost: $90.00
   Budget: 45.0% used
```

---

## 🎯 Single Task (Quick Test)

```bash
# Test on 1 task first
conda run -n wan2.2 python3 run_self_tuning.py \
  --task "G1 humanoid picks up bottle" \
  --task-type g1 \
  --image ./images/g1_workspace_1.jpg \
  --model opus \
  --max-iterations 5 \
  --cost-budget 5

# Expected output:
# - 5 iterations (or early stop if 80+ reached)
# - Cost: ~$2
# - Time: ~30 minutes
# - Video: ./results/raw/g1_XXX/iteration_N/output.mp4
```

---

## 🔄 Checkpoint/Resume (Single Task)

```bash
# Run crashes at iteration 3
conda run -n wan2.2 python3 run_self_tuning.py \
  --task "Task description" \
  --task-type g1 \
  --image workspace.jpg \
  --model opus

# Resume from checkpoint
conda run -n wan2.2 python3 run_self_tuning.py \
  --resume-from ./results/raw/g1_1234567890

# Continues from iteration 4 with full history!
```

---

## 💰 Cost Estimates

| Model | Cost per Task | 100 Tasks | Quality |
|-------|---------------|-----------|---------|
| Claude Opus | $2.00 | $200 | Best (87/100 avg) |
| Claude Sonnet | $0.40 | $40 | Good (84/100 avg) |
| Llama 3.1 70B | $0.00 | $0 | OK (81/100 avg) |
| Qwen 2.5 72B | $0.00 | $0 | OK (80/100 avg) |

---

## 🧪 Recommended Workflow for IROS 2026

### Phase 1: Pilot (Tonight/Tomorrow Morning)

```bash
# Test on 5 tasks (2.5 hours)
# - 2 G1 manipulation
# - 1 UR3 manipulation
# - 1 drone navigation
# - 1 ground navigation

# Verify everything works
# Check learning curves look good
```

### Phase 2: Full Experiment (Tomorrow)

```bash
# Run all 100 tasks with Claude Opus
conda run -n wan2.2 python3 run_batch_experiments.py \
  --tasks my_100_tasks.json \
  --model opus \
  --cost-budget 200

# Runtime: 40-50 hours
# Let it run overnight/unattended
# Checkpoints protect against crashes
```

### Phase 3: Baselines (Next Week)

```bash
# Compare with free models
conda run -n wan2.2 python3 run_batch_experiments.py \
  --tasks my_100_tasks.json \
  --model llama

conda run -n wan2.2 python3 run_batch_experiments.py \
  --tasks my_100_tasks.json \
  --model qwen

# Manual baseline: Run with --max-iterations 1 (single-shot)
```

### Phase 4: Analysis

```bash
# Results in: ./results/iros2026/batch_XXX/final_results.json

# Generate learning curves:
# - Use notebooks/02_visualize_learning_curves.ipynb
# - Plot: Score vs Iteration (Opus vs Llama vs Qwen)
# - Key figure for IROS paper

# Calculate statistics:
# - Success rate: % reaching 80+ in ≤5 iterations
# - Average final score
# - Cost efficiency
```

---

## 🚨 Troubleshooting

### GPU Out of Memory

```bash
# Cosmos 14B too large for 5090? Use 2B:
# Edit src/agentllm_interface.py line 234
# Change: --model-name 14B → --model-name 2B
```

### Ollama Server Down

```bash
# For Llama/Qwen models
curl ${OLLAMA_SERVER:-http://localhost:11434}/health

# Should return: {"status": "ok"}
# If not, restart Ollama server
```

### Invalid API Key

```bash
# Check your key
echo $ANTHROPIC_API_KEY

# Get new key from: https://console.anthropic.com/
export ANTHROPIC_API_KEY="sk-ant-..."
```

### Checkpoint Corruption

```bash
# Remove checkpoint and restart
rm ./results/raw/g1_XXX/checkpoint.json

# Then re-run (will start from iteration 1)
```

---

## 📝 File Outputs

After running, you'll have:

```
results/
├── batch_XXX/
│   ├── final_results.json        # Summary: success rate, cost, etc.
│   ├── batch_checkpoint.json     # Resume point
│   └── tasks/
│       ├── g1_YYY/               # Task 1 output
│       │   ├── checkpoint.json
│       │   ├── iteration_1/
│       │   │   ├── prompts.json
│       │   │   ├── output.mp4
│       │   │   ├── validation.json
│       │   │   └── optimization.json
│       │   ├── iteration_2/
│       │   └── ...
│       ├── ur3_ZZZ/              # Task 2 output
│       └── ...
```

---

## 🎯 Success Criteria for IROS 2026

- ✅ 100 tasks completed (50 manip + 50 nav)
- ✅ 80-85% success rate (reach 80+ score)
- ✅ Learning curves show improvement
- ✅ Claude Opus > Llama/Qwen (statistical significance)
- ✅ Cost: ~$200 total
- ✅ Ready for paper figures and tables

---

**Good luck with your 100-task experiments tomorrow! 🚀**

Questions? Check:
- [README.md](README.md) — Full documentation
- [CHANGELOG.md](CHANGELOG.md) — What's new in v2.1
