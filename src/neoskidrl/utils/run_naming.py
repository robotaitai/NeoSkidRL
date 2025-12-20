"""Run naming utilities for experiment tracking."""
from __future__ import annotations

import random
from datetime import datetime


# Fun, memorable adjectives and nouns for run names
ADJECTIVES = [
    "swift", "bold", "clever", "eager", "fierce", "graceful", "happy", "keen",
    "lucky", "mighty", "noble", "patient", "quick", "steady", "wise", "zealous",
    "agile", "brave", "calm", "daring", "fleet", "gentle", "nimble", "smart",
    "bright", "cosmic", "dynamic", "electric", "fluent", "golden", "heroic",
    "iron", "jade", "kinetic", "lunar", "mystic", "neural", "omega", "primal",
    "quantum", "radiant", "stellar", "turbo", "ultra", "vital", "wild", "xenon",
]

NOUNS = [
    "falcon", "dragon", "phoenix", "tiger", "wolf", "eagle", "hawk", "lion",
    "panther", "raven", "viper", "cobra", "cheetah", "jaguar", "lynx", "orca",
    "shark", "bear", "fox", "owl", "swift", "condor", "leopard", "puma",
    "atlas", "beacon", "cipher", "dagger", "echo", "fusion", "ghost", "horizon",
    "iris", "jade", "karma", "laser", "matrix", "nexus", "orbit", "prism",
    "quest", "rocket", "spark", "titan", "vector", "wave", "zenith", "apex",
]


def generate_run_name(prefix: str = "", suffix: str = "", use_timestamp: bool = True) -> str:
    """
    Generate a memorable run name with optional timestamp.
    
    Args:
        prefix: Optional prefix (e.g., "sac", "ppo")
        suffix: Optional suffix (e.g., "experiment1")
        use_timestamp: Whether to include timestamp (default: True)
    
    Returns:
        Run name in format: prefix_adjective-noun_timestamp_suffix
        Example: "sac_swift-falcon_20231220_143052"
    
    Examples:
        >>> generate_run_name()
        'bold-tiger_20231220_143052'
        
        >>> generate_run_name(prefix="sac")
        'sac_clever-wolf_20231220_143052'
        
        >>> generate_run_name(prefix="sac", suffix="v2", use_timestamp=False)
        'sac_happy-eagle_v2'
    """
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    
    name_parts = []
    
    # Add prefix if provided
    if prefix:
        name_parts.append(prefix)
    
    # Core name
    name_parts.append(f"{adjective}-{noun}")
    
    # Add timestamp if requested
    if use_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_parts.append(timestamp)
    
    # Add suffix if provided
    if suffix:
        name_parts.append(suffix)
    
    return "_".join(name_parts)


def parse_run_name(run_name: str) -> dict:
    """
    Parse a run name into its components.
    
    Args:
        run_name: Run name string
    
    Returns:
        Dictionary with keys: prefix, adjective, noun, timestamp, suffix
    
    Example:
        >>> parse_run_name("sac_swift-falcon_20231220_143052_v2")
        {
            'prefix': 'sac',
            'adjective': 'swift',
            'noun': 'falcon',
            'timestamp': '20231220_143052',
            'suffix': 'v2',
            'full_name': 'sac_swift-falcon_20231220_143052_v2'
        }
    """
    parts = run_name.split("_")
    result = {"full_name": run_name}
    
    # Try to identify timestamp (format: YYYYMMDD_HHMMSS)
    timestamp_idx = None
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():  # Date part
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # Time part
                timestamp_idx = i
                result["timestamp"] = f"{parts[i]}_{parts[i + 1]}"
                break
    
    # Find the adjective-noun part
    name_idx = None
    for i, part in enumerate(parts):
        if "-" in part:
            name_parts = part.split("-", 1)
            if len(name_parts) == 2:
                result["adjective"] = name_parts[0]
                result["noun"] = name_parts[1]
                name_idx = i
                break
    
    # Prefix is everything before the name
    if name_idx is not None and name_idx > 0:
        result["prefix"] = "_".join(parts[:name_idx])
    
    # Suffix is everything after timestamp (or after name if no timestamp)
    if timestamp_idx is not None:
        if timestamp_idx + 2 < len(parts):
            result["suffix"] = "_".join(parts[timestamp_idx + 2:])
    elif name_idx is not None and name_idx + 1 < len(parts):
        # No timestamp, suffix is after name
        if timestamp_idx is None:
            result["suffix"] = "_".join(parts[name_idx + 1:])
    
    return result


if __name__ == "__main__":
    # Test the naming
    print("Generated run names:")
    for _ in range(5):
        print(f"  {generate_run_name(prefix='sac')}")
    
    print("\nParsing example:")
    name = "sac_swift-falcon_20231220_143052_v2"
    parsed = parse_run_name(name)
    print(f"  Name: {name}")
    print(f"  Parsed: {parsed}")

