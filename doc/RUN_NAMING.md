# Run Naming System

## Overview

Every training run now gets a unique, memorable name automatically! No more confusion about which experiment is which.

## Format

Run names follow this pattern:

```
prefix_adjective-noun_timestamp
```

**Example**: `sac_swift-falcon_20231220_143052`

- **prefix**: Algorithm (e.g., "sac", "ppo")
- **adjective-noun**: Memorable name (e.g., "swift-falcon", "bold-tiger")
- **timestamp**: When the run started (YYYYMMDD_HHMMSS)

## Automatic Generation

When you start training without specifying `--run-name`, a random name is generated:

```bash
python -m neoskidrl.scripts.visual_train \
  --config config/recommended_rewards.yml \
  --num-envs 16 \
  --total-steps 300000 \
  --headless
```

Output:
```
============================================================
🚀 Starting training run: sac_swift-falcon_20231220_143052
============================================================

📊 Logging:
  Run ID: sac_swift-falcon_20231220_143052
  Tensorboard: runs/tb/sac_swift-falcon_20231220_143052
  Episodes: runs/metrics/episodes.jsonl
  Checkpoints: runs/checkpoints/sac_swift-falcon_20231220_143052
  Latest: runs/latest/sac_swift-falcon_20231220_143052
```

## Custom Names

You can also specify your own name:

```bash
python -m neoskidrl.scripts.visual_train \
  --run-name my_experiment_v1 \
  --config config/recommended_rewards.yml \
  --headless
```

## Where Names Appear

### 1. Directory Structure
```
runs/
├── tb/
│   ├── sac_swift-falcon_20231220_143052/
│   ├── sac_bold-tiger_20231220_150030/
│   └── sac_clever-wolf_20231220_163245/
├── checkpoints/
│   ├── sac_swift-falcon_20231220_143052/
│   │   ├── ckpt_020000.zip
│   │   ├── ckpt_040000.zip
│   │   └── ...
│   └── ...
├── latest/
│   ├── sac_swift-falcon_20231220_143052.zip
│   └── ...
└── metrics/
    └── episodes.jsonl  # Contains run_id field
```

### 2. Episode Logs

In `runs/metrics/episodes.jsonl`:
```json
{
  "run_id": "sac_swift-falcon_20231220_143052",
  "episode_idx": 42,
  "ep_return": -5.23,
  ...
}
```

### 3. Dashboard

The dashboard can filter by run name:
- Use "Run ID filter" in sidebar
- Type part of the name (e.g., "swift-falcon")
- See only episodes from that run

### 4. Tensorboard

```bash
tensorboard --logdir runs/tb

# Or specific run
tensorboard --logdir runs/tb/sac_swift-falcon_20231220_143052
```

## Available Names

The system has 48 adjectives and 48 nouns, giving **2,304 unique combinations**!

### Adjectives
swift, bold, clever, eager, fierce, graceful, happy, keen, lucky, mighty, noble, patient, quick, steady, wise, zealous, agile, brave, calm, daring, fleet, gentle, nimble, smart, bright, cosmic, dynamic, electric, fluent, golden, heroic, iron, jade, kinetic, lunar, mystic, neural, omega, primal, quantum, radiant, stellar, turbo, ultra, vital, wild, xenon

### Nouns
falcon, dragon, phoenix, tiger, wolf, eagle, hawk, lion, panther, raven, viper, cobra, cheetah, jaguar, lynx, orca, shark, bear, fox, owl, swift, condor, leopard, puma, atlas, beacon, cipher, dagger, echo, fusion, ghost, horizon, iris, jade, karma, laser, matrix, nexus, orbit, prism, quest, rocket, spark, titan, vector, wave, zenith, apex

## Comparison with Other Runs

### Find Your Runs
```bash
# List all runs by date
ls -lt runs/tb/

# Filter by pattern
ls runs/tb/ | grep swift

# Count runs
ls runs/tb/ | wc -l
```

### Compare in Dashboard

1. Launch dashboard
2. Use "Run ID filter" to switch between runs
3. Compare metrics and charts

### Compare in Tensorboard

```bash
# View all runs together
tensorboard --logdir runs/tb

# Select specific runs in the UI
```

## Parsing Run Names

You can programmatically parse run names:

```python
from neoskidrl.utils import parse_run_name

name = "sac_swift-falcon_20231220_143052_v2"
parsed = parse_run_name(name)

print(parsed)
# {
#     'full_name': 'sac_swift-falcon_20231220_143052_v2',
#     'prefix': 'sac',
#     'adjective': 'swift',
#     'noun': 'falcon',
#     'timestamp': '20231220_143052',
#     'suffix': 'v2'
# }
```

## Tips

### Memorable Experiments

If you're running a specific experiment, use a custom name:

```bash
--run-name sac_high_entropy_test
--run-name baseline_v1
--run-name clearance_tuning
```

### Quick Identification

The adjective-noun combination makes it easy to remember:
- "That's the swift-falcon run with the high success rate!"
- "The bold-tiger experiment had too many collisions"

### Chronological Sorting

Timestamps ensure runs are sorted chronologically:
```bash
ls -t runs/tb/  # Most recent first
```

### Clean Up Old Runs

```bash
# Remove specific run
rm -rf runs/tb/sac_old-run_*
rm -rf runs/checkpoints/sac_old-run_*

# Keep only last 10 runs
ls -t runs/tb/ | tail -n +11 | xargs -I {} rm -rf runs/tb/{}
```

## Integration with Existing Tools

### Episode Logger

The run name is automatically used as `run_id` in episode logs, making it easy to filter in the dashboard.

### Evaluation

When evaluating, reference the run by name:

```bash
python -m neoskidrl.scripts.eval \
  --model runs/latest/sac_swift-falcon_20231220_143052 \
  --config config/eval.yml \
  --scenario medium \
  --headless
```

## Examples

### Experiment Series

```bash
# Baseline
python -m neoskidrl.scripts.visual_train \
  --run-name baseline_default_weights \
  --config config/static_goal.yml

# Experiment 1: High progress weight
python -m neoskidrl.scripts.visual_train \
  --run-name exp1_high_progress \
  --config config/recommended_rewards.yml

# Experiment 2: More exploration
python -m neoskidrl.scripts.visual_train \
  --run-name exp2_high_entropy \
  --config config/recommended_rewards.yml \
  --ent-coef 0.2
```

### A/B Testing

```bash
# Version A
python -m neoskidrl.scripts.visual_train \
  --run-name ablation_with_clearance \
  --config config/recommended_rewards.yml

# Version B (edit config to disable clearance)
python -m neoskidrl.scripts.visual_train \
  --run-name ablation_no_clearance \
  --config config/no_clearance.yml
```

## Benefits

1. **No Confusion**: Each run has a unique, memorable name
2. **Easy Comparison**: Filter dashboard by run name
3. **Organized**: Runs are neatly separated in directories
4. **Traceable**: Timestamp shows when each run started
5. **Fun**: Memorable names make experiments more enjoyable!

Happy experimenting! 🚀

