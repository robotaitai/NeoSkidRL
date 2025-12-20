# Reward Dashboard - Quick Start Guide

## 30-Second Setup

```bash
# 1. Install
pip install -e ".[train,ui]"

# 2. Launch dashboard
streamlit run src/neoskidrl/ui/reward_dashboard.py
```

Dashboard opens at: http://localhost:8501

## Start Training (with logging)

```bash
python -m neoskidrl.scripts.visual_train \
  --config config/static_goal.yml \
  --num-envs 16 \
  --total-steps 300000 \
  --chunk-steps 20000 \
  --headless
```

Episodes automatically log to `runs/metrics/episodes.jsonl`

## Dashboard Features

### What You'll See

1. **Top Metrics** (5 cards)
   - Success Rate %
   - Collision Rate %
   - Stuck Rate %
   - Average Return
   - Average Episode Length

2. **Charts**
   - Episode returns over time (line chart)
   - Reward term contributions (stacked area)

3. **Latest Video**
   - Most recent evaluation video from `runs/eval_videos/`

### What You Can Do

**Sidebar Controls:**
- Select config file
- Adjust reward weights (sliders)
- Save changes to config
- Filter by run_id
- Choose last N episodes to display
- Manual refresh

**Auto-refresh:** Every 2 seconds

## Typical Workflow

1. Start training → Dashboard shows real-time progress
2. Notice low success rate → Increase `w_progress` or `w_goal_bonus`
3. Too many collisions → Increase `w_collision` (more negative)
4. Robot too jerky → Increase `w_smooth` (more negative)
5. Save modified config → Restart training
6. Compare results

## Reward Weight Defaults

| Weight | Default | Range | Purpose |
|--------|---------|-------|---------|
| `w_progress` | 1.0 | -10 to 10 | Reward for moving toward goal |
| `w_time` | -0.01 | -1 to 0 | Time penalty per step |
| `w_smooth` | -0.1 | -1 to 0 | Action smoothness penalty |
| `w_collision` | -10.0 | -100 to 0 | Collision penalty |
| `w_goal_bonus` | 10.0 | 0 to 100 | Goal achievement bonus |

## Troubleshooting

**No data showing?**
- Check `runs/metrics/episodes.jsonl` exists
- Verify training is running
- Click "Refresh Now" button

**Config won't save?**
- Check file permissions
- Ensure config path is valid

**Video not displaying?**
- Run evaluation to generate videos
- Check `runs/eval_videos/` has MP4 files

## Tips

- Use **run_id filter** to compare different training runs
- Adjust **last N episodes** to zoom in on recent performance
- **Save As** to create config variants without overwriting
- Watch **stacked area chart** to see which reward terms dominate

## More Info

See `doc/reward_dashboard.md` for complete documentation.

