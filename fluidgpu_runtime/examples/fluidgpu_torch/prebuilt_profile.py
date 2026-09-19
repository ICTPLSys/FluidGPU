from __future__ import annotations

import argparse
import json
from pathlib import Path

from fluidgpu_torch.prebuilt_profiles import prebuilt_profile


def main() -> None:
    args = parse_args()
    profile = prebuilt_profile(args.name, model=args.model)
    args.output_profile.parent.mkdir(parents=True, exist_ok=True)
    args.output_profile.write_text(json.dumps(profile, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Write a prebuilt fluidgpu_torch profile JSON")
    parser.add_argument("--name", required=True)
    parser.add_argument("--output-profile", type=Path, required=True)
    parser.add_argument("--model")
    return parser.parse_args()


if __name__ == "__main__":
    main()
