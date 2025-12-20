# Reward Dashboard

The NeoSkidRL Reward Dashboard is a Streamlit-based UI for monitoring training progress and tuning reward weights in real-time.

## Features

### 1. Real-time Metrics
- **Success Rate**: Percentage of episodes that reached the goal
- **Collision Rate**: Percentage of episodes that ended in collision
- **Stuck Rate**: Percentage of episodes where the robot got stuck
- **Average Return**: Mean episode return over selected episodes
- **Average Episode Length**: Mean number of steps per episode

### 2. Interactive Visualizations
- **Episode Returns Chart**: Line chart showing returns over episodes
- **Reward Term Contributions**: Stacked area chart showing weighted contributions of each reward term (progress, time, smooth, collision, goal_bonus)

### 3. Reward Weight Tuning
Interactive sliders for each reward weight:
- `w_progress`: Reward for making progress toward goal (typically positive)
- `w_time`: Time penalty per step (typically negative)
- `w_smooth`: Smoothness penalty based on action magnitude (typically negative)
- `w_collision`: Collision penalty (typically negative)
- `w_goal_bonus`: Bonus for reaching goal (typically positive)

### 4. Config Management
- **Save**: Update weights in the currently selected config file
- **Save As**: Create a new config file with modified weights
- Supports both legacy format (`w_*` keys) and new format (`reward.weights`)

### 5. Video Display
- Automatically finds and displays the most recent evaluation video
- Shows video path and modification time

### 6. Auto-refresh
- Dashboard updates every 2 seconds automatically
- Manual refresh button available

## Installation

```bash
pip install -e ".[train,ui]"
```

This installs:
- `streamlit>=1.30` - Web UI framework
- `pandas>=2.0` - Data manipulation
- `altair>=5.0` - Declarative visualization
- `streamlit-autorefresh>=1.0.0` - Auto-refresh functionality

## Usage

### Launch Dashboard

```bash
streamlit run src/neoskidrl/ui/reward_dashboard.py
```

Or use the convenience script:

```bash
./run_dashboard.sh
```

The dashboard will open in your browser at `http://localhost:8501`

### During Training

When running training with `visual_train.py`, episode data is automatically logged to `runs/metrics/episodes.jsonl`. The dashboard reads this file and updates in real-time.

Example training command:

```bash
python -m neoskidrl.scripts.visual_train \
  --config config/static_goal.yml \
  --num-envs 16 \
  --total-steps 300000 \
  --chunk-steps 20000 \
  --headless
```

### Filtering Data

Use the sidebar controls to:
- **Run ID filter**: Show only episodes matching a specific run_id pattern
- **Last N episodes**: Limit display to most recent N episodes (default: 100)

## Episode Data Format

Episodes are logged as JSONL (JSON Lines) with one episode per line:

```json
{
  "timestamp": 1703001234.567,
  "run_id": "train_0",
  "algo": "SAC",
  "seed": 0,
  "episode_idx": 42,
  "ep_len": 150,
  "ep_return": -5.23,
  "success": false,
  "collision": false,
  "stuck": false,
  "reward_terms_sum": {
    "progress": 3.2,
    "time": -150.0,
    "smooth": -8.5,
    "collision": 0.0,
    "goal_bonus": 0.0
  },
  "timesteps": 42000
}
```

## Architecture

### Components

1. **EpisodeJSONLLogger** (`src/neoskidrl/train/callbacks.py`)
   - Stable-Baselines3 callback that logs episode data
   - Tracks per-environment episode statistics
   - Writes JSONL format for efficient append-only logging

2. **Dashboard UI** (`src/neoskidrl/ui/reward_dashboard.py`)
   - Streamlit app with sidebar controls and main content area
   - Data loading and aggregation functions
   - Config file reading/writing with backward compatibility
   - Chart rendering with Altair

3. **Integration** (`src/neoskidrl/scripts/visual_train.py`)
   - Automatically enables episode logger during training
   - Writes to `runs/metrics/episodes.jsonl`

### Data Flow

```
Training Loop
    ↓
Environment step() returns info dict with reward_terms
    ↓
EpisodeJSONLLogger accumulates episode data
    ↓
On episode done: write line to episodes.jsonl
    ↓
Dashboard reads JSONL file (auto-refresh every 2s)
    ↓
Display metrics and charts in browser
```

## Testing

Run dashboard utility tests:

```bash
python tests/test_dashboard_smoke.py
```

This verifies:
- JSONL loading and parsing
- Metrics computation
- Reward term expansion
- Config weight extraction and updating

## Troubleshooting

### Dashboard shows "No episode data available"
- Check that `runs/metrics/episodes.jsonl` exists
- Verify training is running with the episode logger enabled
- Try the manual refresh button

### Config changes not saving
- Ensure you have write permissions to the config directory
- Check that the config file path is valid
- Look for error messages in the Streamlit UI

### Charts not updating
- Verify auto-refresh is enabled (should see counter in top-right)
- Check browser console for JavaScript errors
- Try manual refresh button

### Video not displaying
- Ensure evaluation has been run and videos exist in `runs/eval_videos/`
- Check that video files are valid MP4 format
- Verify file permissions

## Future Enhancements

Potential improvements:
- Multi-run comparison (overlay multiple runs on charts)
- Hyperparameter correlation analysis
- Export charts as images
- Real-time training curves (loss, Q-values, etc.)
- Episode replay viewer
- A/B testing of reward configurations

