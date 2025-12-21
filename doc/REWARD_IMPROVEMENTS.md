# Reward Design Improvements

## Summary of Changes

Implemented comprehensive reward shaping improvements based on RL navigation best practices to fix the "zero successes" problem and enable effective learning.

## ✅ What Was Improved

### 1. Enhanced Reward Function

**File**: `src/neoskidrl/rewards/skidnav_reward.py`

Added new reward terms:
- **Stuck penalty**: Penalizes robot for getting stuck
- **Clearance reward**: Encourages maintaining safe distance from obstacles based on minimum lidar reading
- **Heading reward**: Rewards turning toward the goal before moving forward
- **Velocity reward**: Rewards distance moved each step (regardless of direction)

Improved existing terms:
- **Smoothness**: Now penalizes action *changes* (Δaction) instead of action magnitude for better control
- All terms now have proper documentation

### 2. Environment Integration

**File**: `src/neoskidrl/envs/skidnav_env.py`

- Integrated min lidar computation for clearance reward
- Pass stuck flag to reward function
- Track applied vs previous applied action for smoothness reward
- Track goal angle for heading reward
- Proper action history management

### 3. Recommended Reward Configuration

**File**: `config/recommended_rewards.yml`

Created a "get it to learn" configuration with properly scaled weights:

```yaml
reward:
  weights:
    progress: 10.0        # 10x stronger (was 1.0) - main dense signal
    goal_bonus: 75.0      # 3.75x stronger (was 20.0) - agent cares about success
    collision: -75.0      # 7.5x stronger (was -10.0) - strong avoidance
    stuck: -25.0          # NEW - prevents freezing
    clearance: -0.5       # NEW - obstacle awareness
    smooth: -0.05         # Small polish (unchanged)
    time: -0.01           # Tiny penalty (unchanged)
```

**Key improvements:**
- Progress is now the dominant dense signal (10x original)
- Goal bonus is large enough to matter (75 vs 20)
- Collision penalty is strong (-75 vs -10)
- Time penalty kept tiny to avoid drowning learning
- Added clearance shaping for smoother navigation

**Success criteria relaxed:**
- `pos_tol_m`: 0.30 (was 0.20)
- `yaw_tol_deg`: 30.0 (was 10.0) - **much more reasonable!**
- `stop_speed_mps`: 0.15 (was 0.10)

### 4. SAC Exploration Improvements

**File**: `src/neoskidrl/scripts/visual_train.py`

Added entropy tuning for better exploration:

```python
--ent-coef auto          # Default: automatic entropy tuning
--ent-coef 0.1           # Or fix to encourage exploration
--target-entropy -1.0    # Less negative = more exploration
```

### 5. Updated Dashboard

**File**: `src/neoskidrl/ui/reward_dashboard.py`

- Added sliders for `stuck` and `clearance` terms
- Updated tooltips with recommended ranges
- Adjusted default values to match recommended config
- Enhanced explanations

### 6. Bird's Eye View Camera

**File**: `src/neoskidrl/models/omnicar_skid.xml`

Added `bev` camera for top-down visualization:
```xml
<camera name="bev" mode="trackcom" pos="0 0 5" xyaxes="1 0 0 0 1 0"/>
```

## 🚀 Quick Start with Improved Rewards

### Option 1: Use Recommended Config

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

### Option 2: Update Existing Config

If you prefer to modify your existing config, update the weights in `config/static_goal.yml`:

```yaml
reward:
  weights:
    progress: 10.0      # Increase from 1.0
    heading: 2.0        # NEW - turn toward goal
    velocity: 5.0       # NEW - reward moving distance per step
    goal_bonus: 75.0    # Increase from 20.0
    collision: -75.0    # Increase from -10.0
    stuck: -25.0        # Add new term
    clearance: -0.5     # Add new term
    smooth: -0.05       # Keep as is
    time: -0.01         # Keep as is

task:
  success:
    yaw_tol_deg: 30.0   # Relax from 10.0
```

## 📊 Monitoring with Dashboard

Launch dashboard to see the new reward terms:

```bash
streamlit run src/neoskidrl/ui/reward_dashboard.py
```

The pie chart will now show 7 terms including stuck and clearance.

## 🎥 BEV Videos

Generate bird's eye view videos:

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

## 🔍 What to Expect

With these improvements, you should see:

1. **Early successes** (within first 50k-100k steps)
   - With old weights: 0% success rate
   - With new weights: Should see first successes early

2. **Better exploration**
   - Robot should move more confidently toward goal
   - Less "frozen" behavior from overly cautious collision avoidance

3. **Smoother learning curves**
   - Returns should trend upward consistently
   - Success rate should climb steadily

4. **Dashboard insights**
   - Progress term should dominate early (40-50% in pie chart)
   - Goal bonus becomes larger as successes increase
   - Collision/stuck should be small (<10% each) if working well

## 🎯 Tuning After Initial Success

Once you achieve >30% success rate, consider:

1. **Reduce goal bonus**: 75 → 40-50
2. **Increase smoothness**: -0.05 → -0.1 or -0.2
3. **Tighten success criteria**:
   - `pos_tol_m`: 0.30 → 0.20
   - `yaw_tol_deg`: 30 → 20 or 15
4. **Increase clearance**: -0.5 → -1.0 for safer navigation

## 📝 Debugging Checklist

If still getting zero successes after 100k steps:

1. **Check logs** for success conditions:
   ```python
   # Look in episode logger output
   cat runs/metrics/episodes.jsonl | tail -20
   ```

2. **Verify reward terms** in dashboard:
   - Is progress term positive on average?
   - Are collision penalties dominating?
   - Is goal bonus appearing at all?

3. **Watch evaluation videos**:
   ```bash
   python -m neoskidrl.scripts.eval \
     --model runs/latest \
     --config config/eval.yml \
     --scenario easy \
     --episodes 1 \
     --headless
   ```

4. **Check success criteria** aren't too strict:
   - Robot reaching goal but not "succeeding"?
   - Check yaw tolerance (should be ≥20°)
   - Check position tolerance (should be ≥0.25m)

5. **Verify exploration**:
   - Check tensorboard for entropy coefficient
   - Should be >0.01 for v_w actions
   - If too low, use `--ent-coef 0.1`

## 🎓 Reward Design Principles Applied

1. **Progress dominates**: Dense signal is 10x stronger than penalties
2. **Sparse bonus matters**: Goal bonus (75) >> time penalty (-0.01 per step)
3. **Failures are clear**: Collision and stuck are strong and terminal
4. **Shaping is gentle**: Clearance and smoothness guide but don't dominate
5. **Success is achievable**: Relaxed tolerances let agent learn incrementally

## References

Based on best practices from:
- SAC paper entropy tuning recommendations
- OpenAI Spinning Up reward shaping guidelines
- Stable-Baselines3 navigation examples
- Community experience with sparse reward problems

## Files Changed

- `src/neoskidrl/rewards/skidnav_reward.py` - Enhanced reward terms
- `src/neoskidrl/envs/skidnav_env.py` - Integration with environment
- `src/neoskidrl/scripts/visual_train.py` - SAC entropy tuning
- `src/neoskidrl/ui/reward_dashboard.py` - New terms in UI
- `src/neoskidrl/models/omnicar_skid.xml` - BEV camera
- `config/recommended_rewards.yml` - NEW recommended config

All changes are backward compatible with existing configs!
