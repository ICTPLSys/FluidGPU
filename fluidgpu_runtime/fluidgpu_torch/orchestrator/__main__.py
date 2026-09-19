from __future__ import annotations

import argparse

from .run_spec import main as run_main


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m fluidgpu_torch.orchestrator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("spec")
    run_parser.add_argument("--only")
    run_parser.add_argument("--repeat", type=int)
    args = parser.parse_args()
    if args.command == "run":
        run_main(args)


if __name__ == "__main__":
    main()
