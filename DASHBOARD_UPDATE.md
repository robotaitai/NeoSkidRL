# Dashboard Update Summary

## New Features Added

### 1. **New Reward Terms Support**
The dashboard now fully supports all new reward terms from your updated reward function:
- ✅ `progress` - Distance moved toward goal
- ✅ `heading` - Turning toward goal direction  
- ✅ `velocity` - Speed toward goal when far
- ✅ `near_goal_speed` - Penalty for speed near goal (encourages slowing down)
- ✅ `time` - Time penalty per timestep
- ✅ `smooth` - Action smoothness penalty
- ✅ `collision` - Collision penalty
- ✅ `goal_bonus` - Success reward
- ✅ `stuck` - Stuck penalty
- ✅ `clearance` - Too close to obstacles penalty

### 2. **Saved Runs Selection**
- New dropdown in sidebar: **"📦 Load Saved Run"**
- Automatically discovers all saved checkpoints from:
  - `runs/checkpoints/*/*.zip`
  - `runs/latest/*.zip`
  - `runs/final/*.zip`
- Shows 20 most recent runs sorted by modification time
- When you select a saved run:
  - Automatically fills in the run_id filter
  - Shows run info: "🔍 Viewing: run_name"
  - Dashboard displays metrics for that specific run

### 3. **Proper Reward Calculation**
- Now uses `compute_reward_contributions()` and `_resolve_reward_weights()` from `neoskidrl.rewards`
- Respects `enabled_terms` from config
- Only shows and calculates contributions for enabled terms
- More accurate reward breakdown in pie chart and area chart

### 4. **Enhanced Reward Breakdown**
The pie chart now shows:
- **Percentage**: Relative contribution of each term
- **Abs Value**: Absolute value of contribution (for comparison)
- **Raw Value**: Signed contribution (shows positive/negative)
- Only displays **enabled** reward terms from config

### 5. **Updated Weight Ranges**
Sliders now have better default ranges matching your recommended values:
```yaml
progress: 15.0      (range: 0-20)
heading: 2.0        (range: 0-5)
velocity: 0.4       (range: 0-2)
near_goal_speed: -1.0 (range: -5 to 0)
time: -0.005        (range: -1 to 0)
smooth: 0.0         (range: -1 to 0)
collision: -50.0    (range: -100 to 0)
goal_bonus: 100.0   (range: 0-150)
stuck: -20.0        (range: -50 to 0)
clearance: 0.0      (range: -5 to 0)
```

## How to Use

### View a Saved Run
1. Start the dashboard: `./run_dashboard.sh`
2. In the sidebar, look for "📦 Load Saved Run"
3. Select a run from the dropdown (e.g., `runs/checkpoints/sac_brave-falcon_20251221_120000/step_100000.zip`)
4. The dashboard will automatically:
   - Filter episodes for that run_id
   - Display metrics for that specific run
   - Load the config used for that run

### Analyze Reward Contributions
1. The **🥧 Reward Term Breakdown** pie chart shows overall percentage contribution
2. Click "📊 Detailed Breakdown" to see:
   - Percentage of total reward
   - Absolute value (for magnitude comparison)
   - Raw value (signed, shows positive/negative)
3. The **🎨 Reward Term Contributions Over Time** area chart shows how rewards evolve during training
4. Only **enabled** terms from your config are shown

### Adjust Weights
1. Use the sliders in the sidebar to tune each reward term
2. Hover over the ⓘ icon to see recommendations
3. Click "💾 Save weights to config" to update the current config
4. Or "📝 Save as new config copy" to create a new variant

### Compare Runs
1. Select different runs from the "Load Saved Run" dropdown
2. Observe how metrics and reward contributions differ
3. Use this to decide which checkpoint to resume/finetune from

## Technical Details

### Integration with Training
Your updated `visual_train.py` now supports:
- `--resume` - Continue training from a checkpoint (same run_id)
- `--finetune-from` - Start a new run from a checkpoint (new run_id)
- Rich dashboard with real-time progress
- Periodic evaluation with videos

### Reward Function Integration
The dashboard uses the exact same functions as your environment:
- `_resolve_reward_weights(cfg)` - Extracts weights and enabled_terms
- `compute_reward_contributions(terms, cfg)` - Computes weighted contributions
- This ensures the dashboard matches training exactly

### Run Discovery
The dashboard scans for `.zip` files in:
```
runs/
├── checkpoints/
│   └── <run_name>/
│       ├── step_20000.zip
│       ├── step_40000.zip
│       └── ...
├── latest/
│   └── <run_name>.zip
└── final/
    └── <run_name>.zip
```

## Example Workflow

### 1. Start a training run
```bash
python -m neoskidrl.scripts.visual_train \
    --config config/recommended_rewards.yml \
    --total 300000 \
    --chunk 20000 \
    --num-envs 8 \
    --batch-size 512 \
    --eval-every-steps 20000
```

### 2. Monitor in dashboard
```bash
./run_dashboard.sh
```
- Watch metrics update in real-time
- Observe reward term contributions
- See eval videos as they're generated

### 3. Load and compare checkpoints
- Select `runs/checkpoints/sac_brave-falcon_20251221_120000/step_100000.zip`
- Compare with `step_200000.zip`
- See how reward contributions change over training

### 4. Resume or finetune
```bash
# Resume same run
python -m neoskidrl.scripts.visual_train \
    --resume runs/checkpoints/sac_brave-falcon_20251221_120000/step_200000.zip \
    --total 500000

# Finetune with new config
python -m neoskidrl.scripts.visual_train \
    --finetune-from runs/latest/sac_brave-falcon_20251221_120000.zip \
    --config config/new_rewards.yml \
    --total 100000
```

## Files Modified

1. **`src/neoskidrl/ui/reward_dashboard.py`**
   - Added `find_saved_runs()` function
   - Added `extract_run_name_from_path()` function
   - Updated `render_sidebar()` to show saved runs
   - Updated `compute_reward_percentages()` to use proper reward functions
   - Updated `render_reward_terms_chart()` to respect enabled_terms
   - Added support for all 10 reward terms
   - Improved weight ranges and help text

## Testing

To verify everything works:
```bash
# 1. Start dashboard
streamlit run src/neoskidrl/ui/reward_dashboard.py

# 2. Check sidebar shows:
#    - 📦 Load Saved Run dropdown (if checkpoints exist)
#    - All 10 reward term sliders
#    - Updated weight ranges

# 3. Select a saved run and verify:
#    - Run ID filter is auto-filled
#    - Metrics update for that run
#    - Reward contributions show only enabled terms

# 4. Run tests
pytest tests/test_dashboard_utils.py -v
```

## Next Steps

1. **Tune rewards**: Use the dashboard to iteratively adjust weights
2. **Compare checkpoints**: Load different steps and see progress
3. **Resume training**: Use `--resume` to continue from best checkpoint
4. **Experiment**: Try `--finetune-from` with different configs

---

**Updated**: 2025-01-21
**Compatible with**: NeoSkidRL reward system v2 (10 reward terms)

