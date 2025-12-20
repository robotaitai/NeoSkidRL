# Quick Command Reference

## 🎯 Training with Improved Rewards

### Recommended (Best Starting Point)
```bash
python -m neoskidrl.scripts.visual_train \
  --config config/recommended_rewards.yml \
  --num-envs 16 \
  --total-steps 300000 \
  --chunk-steps 20000 \
  --batch-size 512 \
  --ent-coef 0.1 \
  --headless
```

**Run name is auto-generated!** Example: `sac_swift-falcon_20231220_143052`

### With Custom Run Name
```bash
python -m neoskidrl.scripts.visual_train \
  --config config/recommended_rewards.yml \
  --run-name my_experiment_v1 \
  --num-envs 16 \
  --total-steps 300000 \
  --headless
```

### With More Exploration (If Robot Too Cautious)
```bash
python -m neoskidrl.scripts.visual_train \
  --config config/recommended_rewards.yml \
  --num-envs 16 \
  --total-steps 300000 \
  --chunk-steps 20000 \
  --batch-size 512 \
  --ent-coef 0.15 \
  --target-entropy -1.0 \
  --headless
```

## 📊 Monitoring

### Launch Dashboard
```bash
streamlit run src/neoskidrl/ui/reward_dashboard.py
```

## 🎬 Evaluation

### Quick Test (1 Episode, Easy)
```bash
python -m neoskidrl.scripts.eval \
  --algo sac \
  --model runs/latest \
  --config config/eval.yml \
  --scenario easy \
  --episodes 1 \
  --headless
```

### Full Eval (5 Episodes, Medium)
```bash
python -m neoskidrl.scripts.eval \
  --algo sac \
  --model runs/latest \
  --config config/eval.yml \
  --scenario medium \
  --episodes 5 \
  --headless
```

### With BEV Camera
```bash
python -m neoskidrl.scripts.eval \
  --algo sac \
  --model runs/latest \
  --config config/eval.yml \
  --scenario easy \
  --episodes 3 \
  --camera bev \
  --headless
```

### Different Camera Views
```bash
# Track camera (default, 3rd person)
--camera track

# BEV camera (top-down)
--camera bev

# Front camera (1st person)
--camera front
```

## 🔧 Debugging

### Check Episode Logs
```bash
# Last 20 episodes
cat runs/metrics/episodes.jsonl | tail -20

# Count successes
grep '"success":true' runs/metrics/episodes.jsonl | wc -l

# View specific fields
cat runs/metrics/episodes.jsonl | jq '{episode: .episode_idx, return: .ep_return, success: .success}'

# Filter by run name
grep '"run_id":"sac_swift-falcon_20231220_143052"' runs/metrics/episodes.jsonl
```

### Find Your Run
```bash
# List all runs
ls -lt runs/tb/
ls -lt runs/checkpoints/

# Your latest run will be at the top!
```

### Watch Tensorboard
```bash
tensorboard --logdir runs/tb
```

## 📝 Config Quick Edits

### Update Weights in Existing Config
```bash
# Edit your config
nano config/static_goal.yml

# Update these values:
progress: 10.0      # Was 1.0
goal_bonus: 75.0    # Was 20.0
collision: -75.0    # Was -10.0
stuck: -25.0        # New term
clearance: -0.5     # New term
yaw_tol_deg: 30.0   # Was 10.0
```

## 🎮 Key Arguments Explained

### Training
- `--num-envs` : Parallel environments (8-32 recommended)
- `--batch-size` : Training batch size (256-1024)
- `--ent-coef` : Exploration (0.1 for more, 0.01 for less, "auto" default)
- `--target-entropy` : Target entropy (-1.0 for more exploration)
- `--headless` : Offscreen rendering (required on servers)

### Evaluation
- `--episodes` : Number of episodes to run
- `--scenario` : easy/medium/hard (from config)
- `--camera` : track/bev/front
- `--stochastic` : Use random actions instead of deterministic

## 💡 Tips

### If Training Slow
- Increase `--num-envs` (try 32 or 64)
- Reduce `--chunk-steps` (10000 instead of 20000)
- Disable visualization (remove `enable_viz` or run headless)

### If No Successes
1. Check dashboard pie chart - is progress term dominant?
2. Lower `goal_bonus` to 50, increase `progress` to 15
3. Try `--ent-coef 0.2` for more exploration
4. Verify success criteria in config (yaw_tol_deg ≥ 30)

### If Too Many Collisions
1. Increase clearance: `-0.5` → `-1.0`
2. Increase collision penalty: `-75` → `-100`
3. Watch BEV video to see behavior

### If Motion Too Jerky
1. Increase smooth penalty: `-0.05` → `-0.2`
2. Enable rate limiting in config
3. Check action history in tensorboard

## 📈 Expected Timeline

- **0-50k steps**: Random exploration, learning basic movements
- **50k-100k steps**: First successes should appear
- **100k-200k steps**: Success rate climbing to 20-40%
- **200k-300k steps**: Success rate 40-60%, smoother paths
- **300k+ steps**: Polish phase, 60-80% success

## 🎯 Success Metrics

**After 100k steps, you should see:**
- At least 1-5% success rate
- Average return trending upward
- Collision rate declining

**After 200k steps:**
- 20-40% success rate
- Consistent goal-directed behavior
- Smooth, confident navigation

## 🔄 Iteration Loop

1. Train for 100k steps
2. Check dashboard metrics
3. Watch eval video
4. Adjust weights based on behavior
5. Retrain
6. Repeat until satisfied

Good luck! 🚀

