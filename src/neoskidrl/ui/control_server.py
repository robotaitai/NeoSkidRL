#!/usr/bin/env python3
"""
NeoSkidRL Control Server

A local web server that provides:
- Training process control (start/stop)
- Live visualization streaming
- Reward config editing
- Real-time metrics via WebSocket

Run with:
    python -m neoskidrl.ui.control_server --port 8080
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, HTMLResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip install -e '.[dashboard]'")
    sys.exit(1)


# ==============================================================================
# Data Models
# ==============================================================================

class TrainingConfig(BaseModel):
    """Training configuration from UI."""
    config_path: str = "config/recommended_rewards.yml"
    total_steps: int = 100000
    num_envs: int = 4
    batch_size: int = 256
    learning_rate: float = 3e-4
    eval_every_steps: int = 10000
    seed: int = 42


class RewardWeights(BaseModel):
    """Reward weights to save."""
    weights: Dict[str, float]
    config_path: str


class TrainingStatus(BaseModel):
    """Current training status."""
    running: bool
    pid: Optional[int] = None
    run_name: Optional[str] = None
    start_time: Optional[float] = None
    total_steps: Optional[int] = None
    current_steps: Optional[int] = None


# ==============================================================================
# Global State
# ==============================================================================

@dataclass
class ServerState:
    """Global server state."""
    training_process: Optional[subprocess.Popen] = None
    training_config: Optional[TrainingConfig] = None
    run_name: Optional[str] = None
    start_time: Optional[float] = None
    websocket_clients: List[WebSocket] = field(default_factory=list)
    metrics_watcher_task: Optional[asyncio.Task] = None
    

state = ServerState()


# ==============================================================================
# FastAPI App
# ==============================================================================

app = FastAPI(
    title="NeoSkidRL Control Server",
    description="Local control panel for RL training",
    version="1.0.0",
)

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# Static Files & Dashboard
# ==============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML."""
    dashboard_path = Path(__file__).parent / "dashboard" / "control_panel.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    
    # Fallback to index.html
    index_path = Path(__file__).parent / "dashboard" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    return HTMLResponse("<h1>Dashboard not found</h1>")


@app.get("/dashboard/{path:path}")
async def serve_static(path: str):
    """Serve static dashboard files."""
    file_path = Path(__file__).parent / "dashboard" / path
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")


# ==============================================================================
# Training Control Endpoints
# ==============================================================================

@app.get("/api/training/status")
async def get_training_status() -> TrainingStatus:
    """Get current training status."""
    running = state.training_process is not None and state.training_process.poll() is None
    
    return TrainingStatus(
        running=running,
        pid=state.training_process.pid if running else None,
        run_name=state.run_name if running else None,
        start_time=state.start_time if running else None,
        total_steps=state.training_config.total_steps if state.training_config else None,
    )


@app.post("/api/training/start")
async def start_training(config: TrainingConfig):
    """Start a new training run."""
    # Check if already running
    if state.training_process is not None and state.training_process.poll() is None:
        raise HTTPException(status_code=400, detail="Training already running")
    
    # Build command
    cmd = [
        sys.executable, "-m", "neoskidrl.scripts.visual_train",
        "--config", config.config_path,
        "--total-steps", str(config.total_steps),
        "--num-envs", str(config.num_envs),
        "--batch-size", str(config.batch_size),
        "--learning-rate", str(config.learning_rate),
        "--eval-every-steps", str(config.eval_every_steps),
        "--seed", str(config.seed),
        "--headless",
    ]
    
    # Start process
    project_root = Path(__file__).parent.parent.parent.parent
    state.training_process = subprocess.Popen(
        cmd,
        cwd=project_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    state.training_config = config
    state.start_time = time.time()
    
    # Extract run name from output (best effort)
    state.run_name = f"training_{int(state.start_time)}"
    
    # Start metrics watcher
    if state.metrics_watcher_task is None or state.metrics_watcher_task.done():
        state.metrics_watcher_task = asyncio.create_task(watch_metrics())
    
    # Broadcast to clients
    await broadcast({"type": "training_started", "pid": state.training_process.pid})
    
    return {"status": "started", "pid": state.training_process.pid}


@app.post("/api/training/stop")
async def stop_training():
    """Stop the current training run."""
    if state.training_process is None or state.training_process.poll() is not None:
        raise HTTPException(status_code=400, detail="No training running")
    
    # Send SIGTERM
    state.training_process.terminate()
    
    # Wait up to 5 seconds
    try:
        state.training_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        state.training_process.kill()
    
    pid = state.training_process.pid
    state.training_process = None
    state.training_config = None
    state.run_name = None
    state.start_time = None
    
    await broadcast({"type": "training_stopped", "pid": pid})
    
    return {"status": "stopped", "pid": pid}


@app.get("/api/training/logs")
async def get_training_logs(lines: int = 100):
    """Get recent training logs."""
    # Read from episodes.jsonl
    log_path = Path("runs/metrics/episodes.jsonl")
    if not log_path.exists():
        return {"logs": []}
    
    with log_path.open() as f:
        all_lines = f.readlines()
        recent = all_lines[-lines:] if len(all_lines) > lines else all_lines
    
    logs = []
    for line in recent:
        try:
            logs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    
    return {"logs": logs}


# ==============================================================================
# Config Endpoints
# ==============================================================================

@app.get("/api/configs")
async def list_configs():
    """List available config files."""
    project_root = Path(__file__).parent.parent.parent.parent
    config_dir = project_root / "config"
    
    configs = []
    if config_dir.exists():
        for f in sorted(config_dir.glob("*.yml")):
            configs.append({
                "path": str(f.relative_to(project_root)),
                "name": f.stem,
            })
    
    return {"configs": configs}


@app.get("/api/config/{config_name}")
async def get_config(config_name: str):
    """Get a config file's contents."""
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "config" / f"{config_name}.yml"
    
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    
    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    return {"path": str(config_path.relative_to(project_root)), "config": config}


@app.post("/api/config/save")
async def save_config(data: RewardWeights):
    """Save reward weights to a config file."""
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / data.config_path
    
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Config not found")
    
    # Load existing config
    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    # Update weights
    if "reward" not in config:
        config["reward"] = {}
    if "weights" not in config["reward"]:
        config["reward"]["weights"] = {}
    
    for key, value in data.weights.items():
        config["reward"]["weights"][key] = value
    
    # Save back
    with config_path.open("w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    await broadcast({"type": "config_saved", "path": data.config_path})
    
    return {"status": "saved", "path": data.config_path}


# ==============================================================================
# Trajectory Endpoints
# ==============================================================================

@app.get("/api/trajectories")
async def list_trajectories():
    """List available trajectory files."""
    project_root = Path(__file__).parent.parent.parent.parent
    traj_dir = project_root / "runs" / "trajectories"
    
    trajectories = []
    if traj_dir.exists():
        for f in sorted(traj_dir.glob("*.json"), reverse=True)[:100]:
            try:
                with f.open() as fp:
                    data = json.load(fp)
                    trajectories.append({
                        "filename": f.name,
                        "run_id": data.get("metadata", {}).get("run_id", "unknown"),
                        "episode_idx": data.get("episode_idx", 0),
                        "outcome": data.get("outcome", "unknown"),
                        "total_reward": data.get("total_reward", 0),
                        "episode_length": data.get("episode_length", 0),
                    })
            except (json.JSONDecodeError, KeyError):
                continue
    
    return {"trajectories": trajectories}


@app.get("/api/trajectory/{filename}")
async def get_trajectory(filename: str):
    """Get a specific trajectory file."""
    project_root = Path(__file__).parent.parent.parent.parent
    traj_path = project_root / "runs" / "trajectories" / filename
    
    if not traj_path.exists():
        raise HTTPException(status_code=404, detail="Trajectory not found")
    
    with traj_path.open() as f:
        return json.load(f)


# ==============================================================================
# Visualization Endpoints
# ==============================================================================

class EvalRequest(BaseModel):
    """Evaluation request parameters."""
    model_path: Optional[str] = None  # If None, use latest
    config_path: str = "config/recommended_rewards.yml"
    episodes: int = 1
    seed: int = 42
    record_video: bool = True


@app.get("/api/models")
async def list_models():
    """List available model checkpoints."""
    project_root = Path(__file__).parent.parent.parent.parent
    models = []
    
    # Check runs/latest
    latest_dir = project_root / "runs" / "latest"
    if latest_dir.exists():
        for f in sorted(latest_dir.glob("*.zip"), reverse=True):
            models.append({
                "path": str(f.relative_to(project_root)),
                "name": f.stem,
                "type": "latest",
            })
    
    # Check runs/checkpoints
    ckpt_dir = project_root / "runs" / "checkpoints"
    if ckpt_dir.exists():
        for run_dir in sorted(ckpt_dir.iterdir(), reverse=True):
            if run_dir.is_dir():
                for f in sorted(run_dir.glob("*.zip"), reverse=True)[:5]:  # Last 5 per run
                    models.append({
                        "path": str(f.relative_to(project_root)),
                        "name": f"{run_dir.name}/{f.stem}",
                        "type": "checkpoint",
                    })
    
    return {"models": models[:20]}  # Limit to 20 most recent


@app.get("/api/videos")
async def list_videos():
    """List available evaluation videos."""
    project_root = Path(__file__).parent.parent.parent.parent
    videos = []
    
    video_dir = project_root / "runs" / "eval_videos"
    if video_dir.exists():
        for mp4 in sorted(video_dir.rglob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)[:20]:
            videos.append({
                "path": str(mp4.relative_to(project_root)),
                "name": mp4.stem,
                "size_mb": round(mp4.stat().st_size / 1024 / 1024, 2),
            })
    
    return {"videos": videos}


@app.get("/api/video/{path:path}")
async def get_video(path: str):
    """Serve a video file."""
    project_root = Path(__file__).parent.parent.parent.parent
    video_path = project_root / path
    
    if not video_path.exists() or not path.endswith(".mp4"):
        raise HTTPException(status_code=404, detail="Video not found")
    
    return FileResponse(video_path, media_type="video/mp4")


@app.post("/api/eval/run")
async def run_evaluation(req: EvalRequest):
    """Run evaluation on a model and optionally record video."""
    project_root = Path(__file__).parent.parent.parent.parent
    
    # Find model path
    if req.model_path:
        model_path = project_root / req.model_path
    else:
        # Find latest model
        latest_dir = project_root / "runs" / "latest"
        if not latest_dir.exists():
            raise HTTPException(status_code=404, detail="No models found. Train first!")
        
        zips = list(latest_dir.glob("*.zip"))
        if not zips:
            raise HTTPException(status_code=404, detail="No models found in runs/latest")
        
        model_path = max(zips, key=lambda p: p.stat().st_mtime)
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found: {model_path}")
    
    # Build eval command
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_dir = project_root / "runs" / "eval_videos" / "dashboard" / timestamp
    
    cmd = [
        sys.executable, "-m", "neoskidrl.scripts.eval",
        "--model", str(model_path),
        "--config", str(project_root / req.config_path),
        "--episodes", str(req.episodes),
        "--seed", str(req.seed),
        "--output-dir", str(video_dir),
    ]
    
    if req.record_video:
        cmd.extend(["--video-dir", str(video_dir), "--record-video"])
    
    # Run evaluation (blocking for now, could be async later)
    await broadcast({"type": "eval_started", "model": str(model_path.name)})
    
    try:
        result = subprocess.run(
            cmd,
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
        )
        
        # Find generated video
        video_path = None
        if req.record_video and video_dir.exists():
            videos = list(video_dir.glob("*.mp4"))
            if videos:
                video_path = str(videos[0].relative_to(project_root))
        
        await broadcast({
            "type": "eval_completed",
            "success": result.returncode == 0,
            "video_path": video_path,
        })
        
        return {
            "status": "completed",
            "return_code": result.returncode,
            "video_path": video_path,
            "stdout": result.stdout[-2000:] if result.stdout else "",  # Last 2000 chars
            "stderr": result.stderr[-1000:] if result.stderr else "",
        }
        
    except subprocess.TimeoutExpired:
        await broadcast({"type": "eval_failed", "error": "Timeout"})
        raise HTTPException(status_code=504, detail="Evaluation timed out")
    except Exception as e:
        await broadcast({"type": "eval_failed", "error": str(e)})
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# WebSocket for Real-time Updates
# ==============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    state.websocket_clients.append(websocket)
    
    try:
        # Send initial status
        status = await get_training_status()
        await websocket.send_json({"type": "status", "data": status.model_dump()})
        
        # Keep connection alive and handle messages
        while True:
            try:
                data = await asyncio.wait_for(websocket.receive_json(), timeout=30)
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in state.websocket_clients:
            state.websocket_clients.remove(websocket)


async def broadcast(message: dict):
    """Broadcast message to all WebSocket clients."""
    for client in state.websocket_clients[:]:  # Copy list to avoid mutation during iteration
        try:
            await client.send_json(message)
        except Exception:
            state.websocket_clients.remove(client)


async def watch_metrics():
    """Watch for new metrics and broadcast to clients."""
    project_root = Path(__file__).parent.parent.parent.parent
    log_path = project_root / "runs" / "metrics" / "episodes.jsonl"
    
    last_size = 0
    last_episode_idx = -1
    
    while state.training_process is not None and state.training_process.poll() is None:
        try:
            if log_path.exists():
                current_size = log_path.stat().st_size
                if current_size > last_size:
                    # New data available
                    with log_path.open() as f:
                        f.seek(last_size)
                        new_lines = f.readlines()
                    
                    for line in new_lines:
                        try:
                            episode = json.loads(line)
                            if episode.get("episode_idx", -1) > last_episode_idx:
                                last_episode_idx = episode["episode_idx"]
                                await broadcast({
                                    "type": "episode",
                                    "data": episode,
                                })
                        except json.JSONDecodeError:
                            continue
                    
                    last_size = current_size
            
            await asyncio.sleep(1)
            
        except Exception as e:
            print(f"Metrics watcher error: {e}")
            await asyncio.sleep(5)


# ==============================================================================
# Main
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="NeoSkidRL Control Server")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    # Change to project root
    project_root = Path(__file__).parent.parent.parent.parent
    os.chdir(project_root)
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║              🚗 NeoSkidRL Control Server                     ║
╠══════════════════════════════════════════════════════════════╣
║  Dashboard:  http://{args.host}:{args.port}                         ║
║  API Docs:   http://{args.host}:{args.port}/docs                    ║
║                                                              ║
║  Features:                                                   ║
║    • Start/stop training from browser                        ║
║    • Edit reward weights                                     ║
║    • View trajectories                                       ║
║    • Real-time metrics via WebSocket                         ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")
    
    uvicorn.run(
        "neoskidrl.ui.control_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
