#!/usr/bin/env python3
"""
Repairs truncated trace.json files from tracing-chrome.

Dioxus desktop apps may call std::process::exit() which skips Rust's drop
handlers, leaving the trace file truncated mid-write. This script finds
the last complete JSON object and properly closes the array.

Usage:
    ./scripts/fix-trace.py [trace.json] [output.json]

    Defaults: trace.json -> trace_fixed.json
"""

import json
import sys
from pathlib import Path


def fix_trace(input_path: str = "./trace.json", output_path: str = None) -> bool:
    """Fix a truncated Chrome trace JSON file."""

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / f"{p.stem}_fixed{p.suffix}")

    try:
        with open(input_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: {input_path} not found")
        return False

    # First, try loading as-is (maybe it's already valid)
    try:
        json.loads(content)
        print(f"✓ {input_path} is already valid JSON")
        return True
    except json.JSONDecodeError:
        pass

    # Find the last complete JSON object (ends with },)
    last_complete = content.rfind('},')
    if last_complete == -1:
        # Try finding just }
        last_complete = content.rfind('}')
        if last_complete == -1:
            print(f"Error: No complete JSON objects found in {input_path}")
            return False

    # Truncate to last complete object and close the array
    fixed = content[:last_complete+1] + '\n]'

    # Verify the fix worked
    try:
        data = json.loads(fixed)
        event_count = len(data)
    except json.JSONDecodeError as e:
        print(f"Error: Could not repair JSON: {e}")
        return False

    # Write the fixed file
    with open(output_path, 'w') as f:
        f.write(fixed)

    print(f"✓ Fixed: {input_path} -> {output_path}")
    print(f"  {event_count} trace events")
    print(f"  Load in chrome://tracing or https://speedscope.app")

    return True


if __name__ == "__main__":
    input_file = sys.argv[1] if len(sys.argv) > 1 else "./trace.json"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    success = fix_trace(input_file, output_file)
    sys.exit(0 if success else 1)
