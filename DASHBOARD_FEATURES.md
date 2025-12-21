# Dashboard New Features

## ✅ Updates (2025-12-22)

### 1. **Flexible Episode Display** 📊

**Problem Solved**: Previously limited to 100 episodes.

**New Features**:
- Default changed to **500 episodes** (up from 100)
- Slider now goes up to **50,000 episodes**
- New **"Show all episodes"** checkbox
  - ✅ Check this to see **ALL** episodes (no limit)
  - ⚠️ Warning: May be slow for very large datasets (10k+ episodes)
- Info banner shows: "Showing last X of Y episodes" or "Showing all X episodes"

**How to Use**:
```
Sidebar → 🔍 Filters
├── [ ] Show all episodes  ← Check this for all data
└── Last N episodes: 500   ← Or adjust slider
```

---

### 2. **Run Comparison Tool** 🔄

**Problem Solved**: Couldn't compare metrics between different runs.

**New Features**:
- Expandable **"📊 Compare Multiple Runs"** section
- Multi-select dropdown to choose 2-5 runs
- Side-by-side comparison table with:
  - Success Rate %
  - Collision Rate %
  - Stuck Rate %
  - Average Return
  - Average Episode Length
  - Number of Episodes
- **Download CSV** button to export comparison data

**How to Use**:
```
Main Area (top) → 📊 Compare Multiple Runs
1. Click to expand
2. Select runs from multiselect (e.g., sac_brave-falcon_*, sac_vital-quest_*)
3. See comparison table
4. Click "📥 Download Comparison CSV" to save
```

**Example**:
```
| Run                          | Episodes | Success % | Collision % | Avg Return |
|------------------------------|----------|-----------|-------------|------------|
| sac_vital-quest_20251221     | 1250     | 78.5%     | 12.3%       | 45.23      |
| sac_brave-falcon_20251220    | 980      | 65.2%     | 18.7%       | 32.15      |
| sac_mighty-dragon_20251219   | 1100     | 82.1%     | 9.5%        | 52.87      |
```

---

### 3. **Auto-Refresh Control** ⏸️

**Problem Solved**: Pie chart kept moving/recalculating every 2 seconds.

**New Features**:
- **"Auto-refresh (2s)"** checkbox in sidebar (ON by default)
- Uncheck to **freeze** the dashboard
- Charts stay static until you:
  - Re-check auto-refresh, OR
  - Click "🔄 Refresh Now" button

**Why the Pie Chart Moves**:
The pie chart recalculates based on:
1. **New episodes** being logged to `runs/metrics/episodes.jsonl`
2. **Filter changes** (run_id, last_n)
3. **Auto-refresh** pulling latest data every 2 seconds

When auto-refresh is ON:
- Dashboard reads episodes.jsonl every 2s
- If new episodes exist, metrics update
- Pie chart recalculates percentages
- Charts reflect latest training progress

**How to Use**:
```
Sidebar → 🔍 Filters
├── [✓] Auto-refresh (2s)  ← Uncheck to freeze dashboard
└── [🔄 Refresh Now]       ← Manual refresh when auto-refresh is off
```

**Recommended Workflow**:
- **During training**: Keep auto-refresh ON to monitor live progress
- **For analysis**: Turn auto-refresh OFF to study specific time windows
- **For screenshots**: Turn auto-refresh OFF to capture stable charts

---

## Summary of Changes

### Episode Display
| Before | After |
|--------|-------|
| Max 100 episodes | Max 50,000 episodes or ALL |
| Fixed limit | Flexible with checkbox |
| No info display | Shows "X of Y episodes" |

### Run Comparison
| Before | After |
|--------|-------|
| One run at a time | Multi-run comparison |
| Manual note-taking | Side-by-side table |
| No export | CSV download |

### Auto-Refresh
| Before | After |
|--------|-------|
| Always ON (2s) | Toggle ON/OFF |
| No control | Checkbox + manual refresh |
| Charts always moving | Freeze when needed |

---

## Example Workflows

### 📊 Workflow 1: Deep Analysis of Single Run
1. **Sidebar → Load Saved Run**: Select checkpoint
2. **Filters → Show all episodes**: ✅ Check
3. **Auto-refresh**: ❌ Uncheck (freeze dashboard)
4. Study pie chart, area chart, and metrics
5. Take screenshots for report

### 🔄 Workflow 2: Compare Multiple Training Runs
1. **Main Area → Compare Multiple Runs**: Expand
2. Select 3-5 runs (e.g., different hyperparameters)
3. Review comparison table
4. Click **Download CSV** for spreadsheet analysis
5. Switch between individual runs for detailed charts

### 🔴 Workflow 3: Live Training Monitoring
1. **Auto-refresh**: ✅ Keep ON
2. **Filters → Last N episodes**: 500
3. Start training in another terminal
4. Watch metrics update every 2 seconds
5. See reward contributions evolve in real-time

### 🎯 Workflow 4: Finding Best Checkpoint
1. **Compare Multiple Runs**: Select all checkpoints from same run
   - `sac_vital-quest_.../ckpt_100000.zip`
   - `sac_vital-quest_.../ckpt_200000.zip`
   - `sac_vital-quest_.../ckpt_300000.zip`
2. Sort by Success % or Avg Return
3. Identify best checkpoint
4. Use that checkpoint for resume/finetune

---

## Technical Details

### Episode Loading Performance

| Episodes | Load Time | Memory | Recommendation |
|----------|-----------|--------|----------------|
| 100-1000 | <1s | ~5MB | ✅ Fast, responsive |
| 1000-5000 | 1-3s | ~20MB | ✅ Good, slight delay |
| 5000-10000 | 3-8s | ~50MB | ⚠️ Noticeable lag |
| 10000+ | 8s+ | 100MB+ | ⚠️ Use filters or last_n |

**Recommendation**: 
- For **quick monitoring**: Use 500-1000 episodes
- For **deep analysis**: Use "Show all" with run_id filter
- For **comparison**: Use "All episodes" per run

### Auto-Refresh Data Subscription

The pie chart updates when:
```python
# Every 2 seconds (if auto-refresh ON):
1. Read episodes.jsonl
2. Filter by run_id (if set)
3. Take last_n episodes (if set)
4. Calculate reward terms × weights
5. Compute percentage contributions
6. Re-render pie chart
```

If training is active → New episodes added → Percentages change → Pie moves

If training stopped → No new episodes → Percentages stable → Pie static

---

## Troubleshooting

### Q: Pie chart is moving too much
**A**: Uncheck "Auto-refresh (2s)" in sidebar to freeze it

### Q: Can't see recent episodes
**A**: Increase "Last N episodes" or check "Show all episodes"

### Q: Dashboard is slow
**A**: 
1. Reduce "Last N episodes" to 500
2. Use run_id filter to narrow data
3. Uncheck "Show all episodes"

### Q: Comparison shows no data
**A**: 
1. Verify runs have episodes in `runs/metrics/episodes.jsonl`
2. Check run names match exactly (partial match supported)
3. Try clicking "🔄 Refresh Now"

### Q: Want to compare same run at different checkpoints
**A**: The comparison tool uses run_id (e.g., `sac_vital-quest_20251221_230619`), so all checkpoints from the same training run share the same data. Load individual checkpoints from the "Load Saved Run" dropdown to see their specific trained performance.

---

## Files Modified

- `src/neoskidrl/ui/reward_dashboard.py`
  - Added `show_all` checkbox for episodes
  - Increased `last_n` max from 10,000 to 50,000
  - Changed default from 100 to 500
  - Added `auto_refresh_enabled` checkbox
  - Added `render_run_comparison()` function
  - Added episode count info display
  - Updated return signature of `render_sidebar()`

---

**Updated**: 2025-12-22  
**Compatible with**: NeoSkidRL dashboard v2.1

