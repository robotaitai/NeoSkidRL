#!/usr/bin/env python3
"""
Serve the NeoSkidRL dashboard.

This script:
1. Starts a simple HTTP server for the HTML dashboard
2. Provides an API endpoint for listing trajectory files
3. Opens the dashboard in the default browser
"""

import argparse
import http.server
import json
import os
import socketserver
import webbrowser
from pathlib import Path
from urllib.parse import parse_qs, urlparse


class DashboardHandler(http.server.SimpleHTTPRequestHandler):
    """Custom handler that serves the dashboard and provides API endpoints."""
    
    def __init__(self, *args, trajectories_dir: Path, **kwargs):
        self.trajectories_dir = trajectories_dir
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        parsed = urlparse(self.path)
        
        # API: List trajectory files
        if parsed.path == "/api/trajectories":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            
            trajectories = []
            if self.trajectories_dir.exists():
                for f in sorted(self.trajectories_dir.glob("*.json"), reverse=True)[:100]:
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
            
            self.wfile.write(json.dumps(trajectories).encode())
            return
        
        # API: Get specific trajectory
        if parsed.path.startswith("/api/trajectory/"):
            filename = parsed.path.split("/")[-1]
            filepath = self.trajectories_dir / filename
            
            if filepath.exists() and filepath.suffix == ".json":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                
                with filepath.open() as f:
                    self.wfile.write(f.read().encode())
                return
            else:
                self.send_error(404, "Trajectory not found")
                return
        
        # Serve static files
        super().do_GET()
    
    def log_message(self, format, *args):
        """Suppress logging for cleaner output."""
        pass


def make_handler(trajectories_dir: Path):
    """Factory to create handler with trajectories_dir bound."""
    def handler(*args, **kwargs):
        return DashboardHandler(*args, trajectories_dir=trajectories_dir, **kwargs)
    return handler


def main():
    parser = argparse.ArgumentParser(description="Serve the NeoSkidRL dashboard")
    parser.add_argument("--port", type=int, default=8080, help="Port to serve on (default: 8080)")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser automatically")
    parser.add_argument("--trajectories-dir", type=str, default="runs/trajectories",
                        help="Directory containing trajectory JSON files")
    args = parser.parse_args()
    
    # Change to dashboard directory
    dashboard_dir = Path(__file__).parent / "dashboard"
    if not dashboard_dir.exists():
        print(f"Error: Dashboard directory not found at {dashboard_dir}")
        return 1
    
    os.chdir(dashboard_dir)
    
    # Resolve trajectories directory
    trajectories_dir = Path(args.trajectories_dir).resolve()
    print(f"📁 Trajectories directory: {trajectories_dir}")
    
    # Create server
    handler = make_handler(trajectories_dir)
    
    with socketserver.TCPServer(("", args.port), handler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"\n🚀 NeoSkidRL Dashboard")
        print(f"   URL: {url}")
        print(f"   Press Ctrl+C to stop\n")
        
        if not args.no_browser:
            webbrowser.open(url)
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\n👋 Shutting down...")
            return 0


if __name__ == "__main__":
    exit(main() or 0)
